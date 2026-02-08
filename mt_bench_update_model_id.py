import json
# This script updates the model_id field in a jsonl file containing model outputs.
input_file = '/runs/mt_bench/model_answer/deepseek__deepseek-r1.jsonl'
output_file = '/runs/mt_bench/model_answer/deepseek-r1.jsonl'


new_model_id = 'deepseek-r1'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        try:
            
            data = json.loads(line)
            
            
            if 'model_id' in data:
                data['model_id'] = new_model_id
                data['question_id'] = int(data['question_id'])  

            
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
        except json.JSONDecodeError as e:
            print(f"Can't parse line: {line.strip()}\n error: {e}")

print(f"Done! Modified file saved to {output_file}")