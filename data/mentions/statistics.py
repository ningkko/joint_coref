import json
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import collections

def add_non_singleton_to_list(mention_list, output_list):
    for mention in mention_list:
        if not mention["is_singleton"]:
            output_list.append(mention)
        else:
            pass
    return


def join_non_singleton_entity_mentions():
    with open("ECB_Dev_Entity_gold_mentions.json","r") as file:
        dev_en = json.load(file)
    with open("ECB_Test_Entity_gold_mentions.json","r") as file:
        test_en = json.load(file)    
    with open("ECB_Train_Entity_gold_mentions.json","r") as file:
        train_en = json.load(file)    
    en_li = [dev_en,test_en,train_en]

    total_en = []
    for en in en_li:
        add_non_singleton_to_list(en, total_en)

    with open("total_entity.json","w") as file:
        json.dump(total_en, file, indent=4)


def join_non_singleton_event_mentions():

    with open("ECB_Dev_Event_gold_mentions.json","r") as file:
        dev_ev = json.load(file)
    with open("ECB_Test_Event_gold_mentions.json","r") as file:
        test_ev = json.load(file)    
    with open("ECB_Train_Event_gold_mentions.json","r") as file:
        train_ev = json.load(file)    

    ev_li = [dev_ev,test_ev,train_ev]

    total_ev = []
    for ev in ev_li:
        add_non_singleton_to_list(ev, total_ev)

    with open("total_event.json","w") as file:
        json.dump(total_ev, file, indent=4)

def count_entity(keys, mentions, df):

    for m in mentions:
        _key = m["doc_id"]+"_"+str(m["sent_id"])
        if _key not in keys:
            keys.append(_key)

    for k in keys:
        for m in mentions:
            # print("%s %s"%(k,m["doc_id"]+"_"+str(m["sent_id"])))
            if m["doc_id"]+"_"+str(m["sent_id"]) == k:
                if k in df.index:
                    df.loc[k]["entity"] = df.loc[k]["entity"]+1
                else:
                    row = pd.Series({'entity':0,'event':0},name=k)
                    df = df.append(row)
    return df

def count_event(keys, mentions, df):

    for m in mentions:
        _key = m["doc_id"]+"_"+str(m["sent_id"])
        if _key not in keys:
            keys.append(_key)

    for k in keys:
        for m in mentions:
            if m["doc_id"]+"_"+str(m["sent_id"]) == k:
                if k in df.index:
                    df.loc[k]["event"] = df.loc[k]["event"]+1
                else:
                    row = pd.Series({'entity':0,'event':0},name=k)
                    df = df.append(row)

    return df

def plot_event_dist(df):
    event_freq = collections.Counter(df["event"])
    event_freq = pd.DataFrame.from_dict(event_freq, orient='index').reset_index()
    event_freq.columns = ["occurrence","frequency"]
    event_freq = event_freq.sort_values(by = ["frequency"],ascending=False)

    event_freq.to_csv("event_freq.csv",index=False)

    sns.barplot(x = "occurrence" ,y = "frequency", data = event_freq)
    plt.savefig("event_freq.png")

def plot_entity_dist(df):
    entity_freq = collections.Counter(df["entity"])
    entity_freq = pd.DataFrame.from_dict(entity_freq, orient='index').reset_index()
    entity_freq.columns = ["occurrence","frequency"]
    entity_freq = entity_freq.sort_values(by = ["frequency"],ascending=False)

    entity_freq.to_csv("entity_freq.csv",index=False)

    sns.barplot(x = "occurrence" ,y = "frequency", data = entity_freq)
    plt.savefig("entity_freq.png")


def main():

    # join_non_singleton_event_mentions()
    # join_non_singleton_entity_mentions()
    # with open("total_entity.json","r") as file:
    #     entities = json.load(file)    
    # with open("total_event.json","r") as file:
    #     events = json.load(file)    

    # keys = []
    # df = pd.DataFrame()
    # df = count_entity(keys,entities,df)
    # df = count_event(keys,events,df)
    # df["entity"] = df["entity"].astype(int)
    # df["event"] = df["event"].astype(int)
    # df = df.sort_values(by=["entity","event"],ascending=False)
    # df.to_csv("stats.csv")

    df = pd.read_csv("stats.csv")

    plot_entity_dist(df)
    plot_event_dist(df)
    
main()


