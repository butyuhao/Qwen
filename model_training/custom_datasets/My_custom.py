import json
from datasets import load_dataset, Dataset
My_custom_data = {
"_brainstorming_en":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Brainstorming_en.jsonl",
"_brainstorming_zh":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Brainstorming_zh.jsonl",
"_code_en":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Code_en.jsonl",
"_code_zh":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Code_zh.jsonl",
"_complex_instruction_en":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Complex-Instruction_en.jsonl",
"_complex_instruction_zh":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Complex-Instruction_zh.jsonl",
"_continue_en":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Continue_en.jsonl",
"_continue_zh":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Continue_zh.jsonl",
"_harmless_en":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Harmless_en.jsonl",
"_harmless_zh":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Harmless_zh.jsonl",
"_mix_gpt_4":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/MIX_GPT-4.jsonl",
"_role_playing_en":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Role-Playing_en.jsonl",
"_role_playing_zh":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Role-Playing_zh.jsonl",
"_switching_en":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Switching_en.jsonl",
"_switching_zh":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Switching_zh.jsonl",
"_writing_en":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Writing_en.jsonl",
"_writing_zh":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/Writing_zh.jsonl",
"_lima_chat":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/lima_chat.jsonl",
"_lima_qa":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/lima_qa.jsonl",
"_ruozhiba":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/ruozhiba.jsonl",
"_sharegpt_format":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/HighQuality/shareGPT_format.jsonl",
"_empdia_ly_07_17":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/EmpDia_ly_07_17.jsonl",
"_composition_hq":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/composition_hq.jsonl",
"_composition_inst":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/composition_inst.jsonl",
"_compositions":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/compositions.jsonl",
"_open_domain_subject":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/open_domain&subject.jsonl",
"_psy_diagnose":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/psy_diagnose.jsonl",
"_psy_diagnose_wo_inner":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/psy_diagnose_wo_inner.jsonl",
"_psy_generated_zyg_6_29":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/psy_generated_zyg_6_29.jsonl",
"_psy_gpt4_merge_format":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/psy_gpt4_merge_format.jsonl",
"_psy_gpt4_wo_merge_format":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/psy_gpt4_wo_merge_format.jsonl",
"_search":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/search.jsonl",
"_similar_question":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/similar_question.jsonl",
"_socrates_psy":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/socrates_psy.jsonl",
"_socrates_psy_wo_inner":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/socrates_psy_wo_inner.jsonl",
"_socrates_teaching":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/socrates_teaching.jsonl",
"_zuowen":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/feature/zuowen.jsonl",
"_ecnu_qa":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/honest/ecnu_qa.jsonl",
"_honest":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/honest/honest.jsonl",
"_mix_belle":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/opensource_data/MIX_BELLE.jsonl",
"_mix_en":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/opensource_data/MIX_EN.jsonl",
"_mix_zh_others":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/opensource_data/MIX_ZH-Others.jsonl",
"_transstyle":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/zeroshot_data/TransStyle.jsonl",
"_composition_review":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/zeroshot_data/composition_review.jsonl",
"_correct":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/zeroshot_data/correct.jsonl",
"_emotion_dialog":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/zeroshot_data/emotion_dialog.jsonl",
"_gushi_chengyu_pre":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/zeroshot_data/gushi_chengyu_pre.jsonl",
"_poem_generate":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/zeroshot_data/poem_generate.jsonl",
"_poem_transform":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/zeroshot_data/poem_transform.jsonl",
"_reading_comprehension_en":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/zeroshot_data/reading_comprehension_en.jsonl",
"_rewriting":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/zeroshot_data/rewriting.jsonl",
"_story_generate":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/zeroshot_data/story_generate.jsonl",
"_summary":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/zeroshot_data/summary.jsonl",
"_writing":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/zeroshot_data/writing.jsonl",
"_writing_cot":"/public1/home/stu51205901008/Open-Assistant/data/custom_data/zeroshot_data/writing_cot.jsonl",

}
class MyCustom(Dataset):
    def __init__(self, mode: str, cache_dir: str = None) -> None:
        self.mode = mode
        self.rows = []
        self.system_prompt = []
        import os
        cache_dir = os.path.join(os.getcwd(),cache_dir)
        # print(cache_dir)
        with open(cache_dir,"r",encoding="utf-8") as f:
            self.rows = f.readlines()
        
        for i in range(len(self.rows)):
            data = json.loads(self.rows[i])
            self.rows[i] = data["data"]
            self.system_prompt.append(data["system_prompt"])
            # print(self.rows[i])


    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        dialogue: list = self.rows[index]
        system_prompt: str = self.system_prompt[index]
        if self.mode == "sft":
            return (dialogue,system_prompt,"<|H|>")
        elif self.mode == "rl":
            return tuple(dialogue[:-1])
    
