
from flask import Flask, render_template, request, json, Response, jsonify,escape
 
import Terry_toolkit as tkit
import gc
import subprocess
 
import os
from fun import *
from db import NodesDb

from pre_classifier import *
app = Flask(__name__)

 

def get_post_data():
    """
    从请求中获取参数
    :return:
    """
    data = {}
    if request.content_type.startswith('application/json'):
        data = request.get_data()
        data = json.loads(data)
    else:
        for key, value in request.form.items():
            if key.endswith('[]'):
                data[key[:-2]] = request.form.getlist(key)
            else:
                data[key] = value
    return data


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/addlabel')
def addlabel():
    DB=NodesDb()
    id,title,content,author,url,label=DB.get_unlabel_nodes()[0]
    # for node in item:
    pre_label=pre(title+"\n "+content)
    item={"id": id, "title": title, "content": content, "author": author,"url": url, "label": label,"pre_label":pre_label}
    return render_template("addlabel.html", **item)



@app.route("/json/addlabel",methods=['GET', 'POST'])
# 自定义限制器覆盖了默认限制器
# @limiter.limit("100/minute, 1/second")
def json_addlabel():
    # #句子
    DB=NodesDb()
    data= get_post_data()
    print('data',data)
    # paragraph = request.args.get('text')
    # previous_line=request.args.get('sentence')
    id = data['id']
    label= data['label']
    l=DB.set_unlabel_nodes(id,label)

    


    return jsonify({"state":l})

 

@app.route('/add/data')
def add_data_page():
    return render_template("add_data.html")



if __name__ == "__main__":
    app.run()