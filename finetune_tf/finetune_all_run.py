
import argparse
import os
import mlflow
import glob
import slack
import json
import requests
import gc


parser = argparse.ArgumentParser()
parser.add_argument("--name",help="config file name")
args = parser.parse_args()

def post_to_slack(message):
    webhook_url = ""
    slack_data = json.dumps({"blocks":message})
    response = requests.post(
        webhook_url, data=slack_data,
        headers={'Content-Type': 'application/json'}
    )
    if response.status_code != 200:
        raise ValueError(
            'Request to slack returned an error %s, the response is:\n%s'
            % (response.status_code, response.text)
        )

if __name__ == "__main__":
    #os.system('python run_seq_cls.py --task paws --config_file %s.json'%(args.name)) 
    os.system("python run_seq_cls.py --task nsmc --config_file %s.json"%(args.name))
    #os.system("python run_seq_cls.py --task kornli --config_file %s.json"%(args.name))
    #os.system("python run_seq_cls.py --task question-pair --config_file %s.json"%(args.name))
    #os.system("python run_seq_cls.py --task korsts --config_file %s.json"%(args.name))
    #os.system("python run_ner.py --task naver-ner --config_file %s.json"%(args.name))
    #os.system("python run_squad.py --task korquad --config_file %s.json"%(args.name))
    name = args.name
    mlflow.set_experiment("electra_finetune_%s"%(name))  
    result_lst = [i for i in glob.glob("./ckpt/*/*.txt") if name in i]
    with mlflow.start_run():
        for file in result_lst:
            with open(file,"r") as f:
                txt = f.read()
                task = file.split("/")[2].replace(name,"")
                metric_value = [result.split("=") for result in txt.split("\n")[:-1]]
                for metric,value in metric_value:
                    if "loss" not in metric:
                        mlflow.log_metric(task+"_"+metric,float(value))
    with open("./slack_message.json","r") as f:
        data = json.load(f)
    data[0]["text"]["text"]= name+data[0]["text"]["text"]
    post_to_slack(data)