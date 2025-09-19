"""
Custom dataset class for loading LightR1-Clean4 dataset from HuggingFace.
"""

import os
import re
from typing import Optional, Any, Dict, List, Union
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, ConcatDataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
import logging
logger = logging.getLogger(__name__)


DAPO_PROMPT_BEGIN = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
DAPO_PROMPT_END = "\n\nRemember to put your answer on its own line after \"Answer:\"."
COT_PROMPT_BEGIN = "One approach to this problem is the section enclosed by <think></think> below:\n"
COT_PROMPT_END = "\nYou can refer to this approach to solve the problem."
TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


class LightR1Dataset(RLHFDataset):
    """
    Custom dataset for loading LightR1-Clean4 dataset from HuggingFace.
    """
    
    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(data_files, tokenizer, config, processor)

        # CCOT configuration
        self.enable_ccot = config.get("enable_ccot", False)
        self.ccot_scheduler = config.get("ccot_scheduler", "window")
        self.add_cot_to_answer = config.get("add_cot_to_answer", False)
        if self.enable_ccot:
            self.cot_ratio_list = self._get_cot_ratio_list()
        
    def add_cot_to_answer_impl(self, prompt: str) -> str:
        # Extract content between <think> and </think> tags
        if prompt.endswith(TEMPLATE_SUFFIX):
            prompt  = prompt[:-len(TEMPLATE_SUFFIX)]
        # logger.info(f"[Rollout] Add cot to answer Original: {prompt}")
        # think_match = re.search(r"<think>(.*?)</think>", prompt, re.DOTALL)
        #think_content = think_match.group(1).strip() if think_match else ""
        
        # logger.info(f"[Rollout] Add cot to answer Think content: {think_content}")
        think_content = ""
        
        # Remove the part between COT_PROMPT_BEGIN and COT_PROMPT_END (including the markers)
        if COT_PROMPT_BEGIN in prompt and COT_PROMPT_END in prompt:
            # Find the positions
            cot_begin_idx = prompt.find(COT_PROMPT_BEGIN)
            cot_end_idx = prompt.find(COT_PROMPT_END) + len(COT_PROMPT_END)
            
            # Remove the entire COT section
            think_content = prompt[cot_begin_idx:cot_end_idx][len(COT_PROMPT_BEGIN):-len(COT_PROMPT_END)][len('<think>'):-len('</think>')]
            prompt = prompt[:cot_begin_idx] + prompt[cot_end_idx:]
        
        prompt = prompt + TEMPLATE_SUFFIX + think_content
        # logger.info(f"[Rollout] Add cot to answer Final: {prompt}")
        return prompt
    

    def get_window_cot_ratio(self, dataset_len: int) -> List[float]:
        # K is the step interval
        group_size = 32  # batch_size / n_generation
        K = 50
        if dataset_len < K * group_size * 6:  # 6: first 3 intervals use ccot, the other half disable ccot
            K = dataset_len // (group_size * 6)
        
        # Initialize the result list
        ratio_list = []
        
        for idx in range(dataset_len):
            # Determine which group this element belongs to
            group_idx = idx // group_size
            # Position within the group (0-31 for full groups)
            pos_in_group = idx % group_size
            
            # Determine the ratio based on group index and position
            if group_idx < K:
                # First K groups: 0.6, 0.4, 0.2, 0.0
                if pos_in_group < group_size * 0.25:
                    ratio = 0.6
                elif pos_in_group < group_size * 0.5:
                    ratio = 0.4
                elif pos_in_group < group_size * 0.75:
                    ratio = 0.2
                else:
                    ratio = 0.0
            elif group_idx < 2 * K:
                # Next K groups: 0.4, 0.2, 0.0, 0.0
                if pos_in_group < group_size * 0.25:
                    ratio = 0.4
                elif pos_in_group < group_size * 0.5:
                    ratio = 0.2
                else:
                    ratio = 0.0
            elif group_idx < 3 * K:
                # Next K groups: 0.2, 0.0, 0.0, 0.0
                if pos_in_group < group_size * 0.25:
                    ratio = 0.2
                else:
                    ratio = 0.0
            else:
                # All remaining elements: 0.0
                ratio = 0.0
            
            ratio_list.append(ratio)
        return ratio_list

    def get_reverse_window_cot_ratio(self, dataset_len: int) -> List[float]:
        # K is the step interval
        group_size = 32  # batch_size / n_generation
        K = 50
        if dataset_len < K * group_size * 6:  # 6: first 3 intervals use ccot, the other half disable ccot
            K = dataset_len // (group_size * 6)
        
        # Initialize the result list
        ratio_list = []
        
        for idx in range(dataset_len):
            # Determine which group this element belongs to
            group_idx = idx // group_size
            # Position within the group (0-31 for full groups)
            pos_in_group = idx % group_size
            
            # Determine the ratio based on group index and position
            if group_idx < K:
                # First K groups: 0.0, 0.0, 0.0, 0.1
                if pos_in_group < group_size * 0.25:
                    ratio = 0.0
                elif pos_in_group < group_size * 0.5:
                    ratio = 0.0
                elif pos_in_group < group_size * 0.75:
                    ratio = 0.0
                else:
                    ratio = 0.1
            elif group_idx < 2 * K:
                # Next K groups: 0.0, 0.1, 0.0, 0.0
                if pos_in_group < group_size * 0.25:
                    ratio = 0.0
                elif pos_in_group < group_size * 0.5:
                    ratio = 0.1
                else:
                    ratio = 0.0
            elif group_idx < 3 * K:
                # Next K groups: 0.2, 0.0, 0.0, 0.0
                if pos_in_group < group_size * 0.25:
                    ratio = 0.2
                else:
                    ratio = 0.0
            else:
                # All remaining elements: 0.0
                ratio = 0.0
            
            ratio_list.append(ratio)
        return ratio_list

    def get_stair_cot_ratio(self, dataset_len: int) -> List[float]:
        target_ratio_set = [0.6, 0.4, 0.2, 0.0]
        ratio_list = []
        for idx in range(dataset_len):
            if idx < dataset_len // 4:
                ratio = target_ratio_set[0]
            elif idx < dataset_len // 2:
                ratio = target_ratio_set[1]
            elif idx < dataset_len // 4 * 3:
                ratio = target_ratio_set[2]
            else:
                ratio = target_ratio_set[3]
            ratio_list.append(ratio)
        return ratio_list

    def get_linear_cot_ratio(self, dataset_len: int) -> List[float]:
        ratio_list = []
        for idx in range(dataset_len):
            ratio = idx / dataset_len
            ratio_list.append(ratio)
        return ratio_list
    
    def get_custom_cot_ratio(self, dataset_len: int) -> List[float]:
        ratio_list = []
        for idx in range(dataset_len):
            if idx < 500:
                ratio = 0.6
            elif idx < 1000:
                ratio = 0.4
            elif idx < 1500:
                ratio = 0.2
            else:
                ratio = 0.0
            ratio_list.append(ratio)
        return ratio_list
    
    def get_reverse_stair_cot_ratio(self, dataset_len: int) -> List[float]:
        ratio_list = []
        for idx in range(dataset_len):
            if idx < int(dataset_len * 0.9):
                ratio = 0.0
            elif idx < int(dataset_len * 0.95):
                ratio = 0.1
            else:
                ratio = 0.2
            ratio_list.append(ratio)
        return ratio_list 


    
    def _get_cot_ratio_list(self) -> List[float]:
        """Generate COT ratio list for curriculum learning."""
        dataset_len = len(self.dataframe)
        if self.ccot_scheduler.lower() == "window":
            return self.get_window_cot_ratio(dataset_len)
        elif self.ccot_scheduler.lower() == "all":
            return [1.0] * dataset_len
        elif self.ccot_scheduler.lower() == "stair":
            return self.get_stair_cot_ratio(dataset_len)
        elif self.ccot_scheduler.lower() == "linear":
            return self.get_linear_cot_ratio(dataset_len)
        elif self.ccot_scheduler.lower() == "reverse_stair":
            return self.get_reverse_stair_cot_ratio(dataset_len)
        elif self.ccot_scheduler.lower() == "reverse_window":
            return self.get_reverse_window_cot_ratio(dataset_len)
        elif self.ccot_scheduler.lower() == "custom":
            return self.get_custom_cot_ratio(dataset_len)
        else:
            raise ValueError(f"Unknown CCOT scheduler: {self.ccot_scheduler}")
    
    def _extract_cot(self, text: str) -> str:
        """Extract content between <think> and </think> tags."""
        pattern = r"<think>(.*?)</think>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1) if match else ""
    
    def _split_cot(self, cot: str, idx: int) -> str:
        """Split COT based on ratio for curriculum learning."""
        ratio = self.cot_ratio_list[idx]
        # print(ratio, idx, '------------')

        # Special case for "all" scheduler
        if "all" in self.ccot_scheduler.lower():
            sentences = cot.split('. ')
            if len(sentences) < 1:
                return ""
            return '. '.join(sentences[:-1])

        if ratio <= 0.0:
            return ""
        else:
            # Split by ". " to get sentences
            sentences = cot.split('. ')
            if len(sentences) < 1:
                return ""
            
            # Calculate number of sentences to keep based on ratio
            num_used_sentences = int(len(sentences) * ratio)
            if num_used_sentences < 1:
                return ""
            
            # Take the first num_used_sentences
            selected_sentences = sentences[:num_used_sentences]
            
            # Join back with ". " and handle the last sentence
            result = '. '.join(selected_sentences)
            
            # If the original cot ended with ". ", preserve it
            # Otherwise, check if we need to add ". " at the end
            if num_used_sentences < len(sentences):
                # We didn't take all sentences, so add ". " at the end
                if not result.endswith('.'):
                    result += '.'
            elif cot.endswith('. '):
                # We took all sentences and original ended with ". "
                if not result.endswith('. '):
                    result += '. '
            
            return result
    
    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Get a single item from the dataset."""
        
        # Create output file path
        # output_file = os.path.join(self.output_dir, f"{item}.txt")
        
        # Get data from HuggingFace dataset
        row_dict = self.dataframe[item]
        # Extract fields
        problem = row_dict.pop(self.prompt_key)[0]['content']
        
        # Build prompt with DAPO format
        content = problem
        prompt = DAPO_PROMPT_BEGIN + content + DAPO_PROMPT_END

        # Handle CCOT if enabled
        if self.enable_ccot:
            solution = row_dict.pop('cot')
            if solution:
                cot = self._extract_cot(solution)
                # print('Raw Cot: ', cot)
                # print('=============')
                cot_used = self._split_cot(cot, item)
                # print('Cot Used: ', cot_used)
                # print('=============')
                if len(cot_used) > 0:
                    cot_used = "<think>" + cot_used + "</think>"
                    prompt = DAPO_PROMPT_BEGIN + content + COT_PROMPT_BEGIN + cot_used + COT_PROMPT_END + DAPO_PROMPT_END
                else:
                    prompt = DAPO_PROMPT_BEGIN + content + DAPO_PROMPT_END

        # Convert to chat format
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Apply chat template
        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        
        if self.enable_ccot and self.add_cot_to_answer:
            raw_prompt = self.add_cot_to_answer_impl(raw_prompt)
        

        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if not self.processor_type == "MiniCPMVImageProcessor":
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )

            if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                from verl.models.transformers.qwen2_vl import get_rope_index

                position_ids = [
                    get_rope_index(
                        self.processor,
                        input_ids=input_ids[0],
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        video_grid_thw=model_inputs.get("video_grid_thw"),
                        second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                        attention_mask=attention_mask[0],
                    )
                ]  # (1, 3, seq_len)

            else:
                position_ids = compute_position_id_with_mask(attention_mask)

            row_dict["input_ids"] = input_ids[0]
            row_dict["attention_mask"] = attention_mask[0]
            row_dict["position_ids"] = position_ids[0]
        else:
            row_dict["input_ids"] = input_ids
            row_dict["attention_mask"] = attention_mask
            row_dict["position_ids"] = position_ids

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict


def main():
    """
    Main function to test LightR1Dataset by printing the first 100 elements to a txt file.
    Configuration is based on recipe/dapo/run_dapo_lightr1_qwen2.5_32b.sh
    """
    import json
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    
    # Setup configuration based on run_dapo_lightr1_qwen2.5_32b.sh
    config = OmegaConf.create({
        "enable_ccot": True,
        "add_cot_to_answer": True,
        "max_prompt_length": 8192,
        "max_response_length": 2048,
        "truncation": "left",
        "prompt_key": "prompt",
        "need_tools_kwargs": False,
        "return_raw_chat": False,
        "return_full_prompt": True,
    })
    
    # Model path from the shell script
    model_path = "Qwen/Qwen2.5-7B"
    
    # Load tokenizer
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Data file path from the shell script
    data_files = "/root/workspace/ccot_verl/data/train.parquet"
    
    # Check if data file exists
    if not os.path.exists(data_files):
        print(f"Warning: Data file {data_files} not found. Using HuggingFace dataset instead.")
        # Use HuggingFace dataset if local file doesn't exist
        data_files = None
    
    # Create dataset instance
    print("Creating LightR1Dataset instance...")
    dataset = LightR1Dataset(
        data_files=data_files,
        tokenizer=tokenizer,
        config=config,
        processor=None
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get first 100 elements (or all if less than 100)
    num_samples = min(100, len(dataset))
    
    # Create output file
    output_file = "lightr1_dataset_first_100_samples.txt"
    
    print(f"Extracting first {num_samples} samples...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"LightR1Dataset - First {num_samples} Samples\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Data file: {data_files}\n")
        f.write(f"Configuration:\n{json.dumps(OmegaConf.to_container(config), indent=2)}\n")
        f.write("="*100 + "\n\n")
        
        for idx in range(num_samples):
            print(f"Processing sample {idx + 1}/{num_samples}...")
            
            try:
                sample = dataset[idx]
                
                f.write(f"Sample {idx}:\n")
                f.write("-"*80 + "\n")
                
                # Write key information from the sample
                f.write(f"{sample}\n")
                
                f.write("\n" + "="*80 + "\n\n")
                
            except Exception as e:
                f.write(f"Error processing sample {idx}: {str(e)}\n")
                f.write("="*80 + "\n\n")
                print(f"Error processing sample {idx}: {str(e)}")
    
    print(f"Successfully saved {num_samples} samples to {output_file}")
    print(f"Additional prompt outputs have been saved to directory")


if __name__ == "__main__":
    main()