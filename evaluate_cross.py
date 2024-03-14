import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import math
import time
import json
from tqdm import tqdm
from multiprocessing import Process
import torch
from vllm import LLM, SamplingParams

@torch.no_grad()
def greedy_decoding(model, tokenizer, input_ids, attention_mask, target_ids, max_new_tokens=64, eos_token_id = None, early_stop = True):
    """
    - k: top-k candidate words are selected, default 3
    - alpha: (1-alpha)p_lm -(alpha)*penalty
    - max_length: decoding max_length-prompt_length steps
    - n: the order of n-gram models
    - target_ids: obtain the probs of target_ids
    - sw_coeff: give stopwords a small penalty (<1) or larger penalty(>1), default 0.
    - stop_words=[]: the list of stopwords. If you use GPT-2, you at least need to add two special tokens ('Ċ' and 'ĊĊ') to avoid grammars errors.
    """
    batch_size, prefix_len = input_ids.size()
    model_kwargs = {}
    prompt_len = torch.sum(attention_mask, dim=1)
    model_kwargs["attention_mask"] = attention_mask

    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

    eos_token_id_tensor = torch.tensor([eos_token_id]).to(model.device) if eos_token_id is not None else None
    unfinished_sequences = input_ids.new(batch_size).fill_(1)

    for step in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        #print("model inputs:",model_inputs)
        outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)
        next_token_scores = outputs.logits[:, -1, :]

        # avoid generating eos
        if not early_stop and eos_token_id != None:
            next_token_scores[:, eos_token_id] = -float("inf")

        if step == 0:
            target_scores = next_token_scores[:,target_ids]
        next_tokens = torch.argmax(next_token_scores, dim=-1)
        # fsd-vec
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (1 - unfinished_sequences)
        #print("next tokens:",next_tokens)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

        if unfinished_sequences.max() == 0 or step == max_new_tokens - 1:
            stopped = True
        else:
            stopped = False

        if stopped:
            break

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
    a,b = torch.topk(target_scores,dim=-1,k=1)
    #print(b)
    return input_ids, b


def args_parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--infile_list', type=str,help='the data used for instructing tuning')
    #parser.add_argument('--infile', type=str, default="",help='the data used for instructing tuning')
    parser.add_argument('--outfile', type=str, help='the data used for instructing tuning')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--gpus_per_model', type=int, default=1)
    parser.add_argument('--model_name_or_path', default="decapoda-research/llama-7b-hf", type=str)


    args = parser.parse_args()
    return args


def out_file(outfile_path, generation_lst):
    with open(outfile_path, 'w', encoding="utf-8") as f:
        json.dump(generation_lst, f, indent=4)

    print(f'written to {outfile_path}')


def generate(args):
    #visible_devices = [str(rank * args.gpus_per_model + i) for i in range(args.gpus_per_model)]
    #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)

    if "summary" in args.infile_list:
        task = "summary"

    if "question" in args.infile_list:
        task = "question"

    if "sentiment" in args.infile_list:
        task = "sentiment"
    if "detection" in args.infile_list:
        task = "detection"
    if "inference" in args.infile_list:
        task = "inference"

    if task == "summary" or task == "question":
        model = LLM(model=args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            # tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            # torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        model.bfloat16()
        model.eval()


    with open("data/prompt_map.json",encoding="utf-8") as f:
        prompt_maps = json.load(f)

    #args.infile_list = ./data/summary/xlsum_pdfs_cnn_xsum
    dir_prefix = "/".join(args.infile_list.split("/")[:-1]) #./data/summary
    files = args.infile_list.split("/")[-1].split("_")
    print("dir_prefix: ",dir_prefix)
    print("file list: ",files)

    for ffff in files:
        infile = dir_prefix+"/"+ffff+"_test_500.json"
        print("current file: ",infile)
        with open(infile) as f:
            prompt_lst = json.load(f)
        # we only test 250 samples
        prompt_lst = prompt_lst[0:250]
        
        fff = dir_prefix+"/few_shot/"+ffff+"_few_shot.json"
        
        with open(fff,encoding="utf-8") as f:
            template = json.load(f)

        if task=="summary" or task == "question":
            shot_list = [4,2,1]
        elif task == "sentiment" or task=="detection":
            shot_list = [2,4,6]
        else:
            shot_list = [3,5,7]

        #if "llama" not in args.model_name_or_path:
        # also test 0-shot performance 
        shot_list.append(0)

        for shot in shot_list:
            for map_idx, map_function in enumerate(prompt_maps):
                print("map function: ",map_function)
                context = ""
                for tmp in template[:shot]:
                    context+=map_function[task].format_map(tmp)+tmp["output"]+"\n\n"
                print(f"the total number of prompts: {len(prompt_lst)}")

                generation_res = []
                s = time.time()
                if task == "summary" or task == "question":
                    sampling_params = SamplingParams(temperature=0, top_p=1,max_tokens=60)
                
                    if shot == 0:
                        prompt_text = ["<s> " + map_function[task].format_map(x) for x in prompt_lst]
                    else:
                        prompt_text = [context + map_function[task].format_map(x) for x in prompt_lst]
                    
                    outputs = model.generate(prompt_text, sampling_params)
                    
                    for ppp, output in enumerate(outputs):
                        prompt = output.prompt
                        generated_text = output.outputs[0].text
                    
                        if task == "question":
                            generation_res.append(
                                {"idx": prompt_lst[ppp]['idx'],
                                 "instruction": prompt_lst[ppp]['input'] + "\n\n" + prompt_lst[ppp]['answer'],
                                 "generation": generated_text.strip(), "output": prompt_lst[ppp]["output"]})
                        else:
                            generation_res.append(
                                {"idx": prompt_lst[ppp]['idx'], "instruction": prompt_lst[ppp]['input'],
                                 "generation": generated_text.strip(), "output": prompt_lst[ppp]["output"]})

                else:
                    for start in tqdm(range(0, len(prompt_lst), args.batch_size)):
                        cur_prompt_lst = prompt_lst[start: start + args.batch_size]
                        if shot == 0:
                            prompt_text = ["<s> " + map_function[task].format_map(x) for x in cur_prompt_lst]
                        else:
                            prompt_text = [context + map_function[task].format_map(x) for x in cur_prompt_lst]
                        if start == 0:
                            print("prompt_text: ",prompt_text)
                        model_inputs = tokenizer(prompt_text, padding=True, add_special_tokens=False, return_tensors="pt")
                        input_ids = model_inputs["input_ids"].to(model.device)
                        attention_mask = model_inputs["attention_mask"].to(model.device)
                        prompt_len = input_ids.size(1)

                        if task == "summary" or task == "question":
                            outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=args.max_new_tokens)

                        if task =="sentiment":
                            outputs, target_scores = greedy_decoding(model, tokenizer, input_ids, attention_mask,
                                                                     target_ids=[1066, 22198],
                                                                     max_new_tokens=args.max_new_tokens)
                        if task == "inference":
                            outputs, target_scores = greedy_decoding(model, tokenizer, input_ids, attention_mask,
                                                                     target_ids=[296, 17821, 9996], #entailment,neutral,contradiction
                                                                     max_new_tokens=args.max_new_tokens)

                        if task == "detection":
                            outputs, target_scores = greedy_decoding(model, tokenizer, input_ids, attention_mask,
                                                                     target_ids=[3582, 1217], #yes no
                                                                     max_new_tokens=args.max_new_tokens)

                        generation_text = tokenizer.batch_decode(outputs[:, prompt_len:], clean_up_tokenization_spaces=True,
                                                                 skip_special_tokens=True)
                        if start==0:
                            print("gend text: ", generation_text)

                        if task == "inference" or task=="sentiment" or task=="detection":
                            target_scores = target_scores.squeeze(1).cpu().tolist()
                            for prompt, generation, ts in zip(cur_prompt_lst, generation_text, target_scores):
                                if task == "sentiment":
                                    generation_res.append(
                                        {"idx": prompt['idx'], "instruction": prompt['input'],
                                         "generation": generation.strip(), "output": prompt["output"], "max_idx": ts})
                                else:
                                    generation_res.append(
                                        {"idx": prompt['idx'],
                                         "instruction": prompt['input_1'] + "\n\n" + prompt['input_2'],
                                         "generation": generation.strip(), "output": prompt["output"],"max_idx":ts})

                        else:
                            for prompt, generation in zip(cur_prompt_lst, generation_text):
                                if task == "question":
                                    generation_res.append(
                                        {"idx": prompt['idx'],
                                         "instruction": prompt['input'] + "\n\n" + prompt['answer'],
                                         "generation": generation.strip(), "output": prompt["output"]})
                                else:
                                    generation_res.append(
                                        {"idx": prompt['idx'], "instruction": prompt['input'],
                                         "generation": generation.strip(), "output": prompt["output"]})

                t = time.time()
                print("time used: ", t - s)
                mmm = args.model_name_or_path.split("/")[-1]
                outfile = args.outfile+ffff+"_"+mmm+"_"+str(shot)+"_shot_map_"+str(map_idx)+".json"
                print(outfile)
                out_file(outfile, generation_res)

if __name__ == "__main__":
    args = args_parse()
    print(args)
    generate(args)



