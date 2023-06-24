from queries import insert_query, select_query
from flask import Blueprint, request, jsonify
from flask import request
from imageio import imread
import pandas as pd
from classifiers import bietapic_1, bietapic_2, opacity_detector
import json
from numpy import exp

ssd = Blueprint("ssd", __name__)


# @ssd.route("/",methods=['POST'])
# def output():
#     img = request.files['image']
#     path = 'images/{}'.format(img.filename)
#     img.save(path)

#     # Insert Query 
#     table = 'images'
#     columns = ["id","path","width","height"]
#     values = ['NULL','']
#     insert_query(table,columns,values)


#     # Insert Query 
#     table = 'images'
#     columns = ["id","path","width","height","thorax","thorax_softmax"]
#     insert_query(table,columns,values)


#     fields = ["id","path"]
#     tables = ["images"]
#     union  = None
#     where = None
#     base = ''
#     result = select_query(fields,tables,union,where,base).to_dict(orient='index')
#     return jsonify(list(result.values()))

@ssd.route("/hola",methods=['POST'])
def output2():
    img = request.files['image']
    path = 'images/{}'.format(img.filename)
    img.save(path)
    size = imread(path).shape
    
    # Image Insert Query 
    table = 'images'
    columns = ["id","path","width","height"]
    values = [None,path,size[0],size[1]]
    insert_query(table,columns,values)

    #df = pd.DataFrame(values,columns=columns)
    chest_softmax,chest = bietapic_1(path)
    Chest = {'chest':str(chest),'chest_softmax':str(chest_softmax)}
    dict = {'Chest':Chest}
    
    if chest == 'Chest':
        view_softmax,view = bietapic_2(path)
        View = {'view':str(view),'PA':str(view_softmax[2]),'AP':str(view_softmax[0]),'L':str(view_softmax[1])}
        dict['View'] = View
    
    df_id = select_query(['MAX(id)'],['images'],None,None,'')
    # Image Insert Query 
    table = 'ssd_results'
    columns = ["id","thorax","thorax_softmax","projection",
    "pa_softmax","ap_softmax","l_softmax",'image_id']
    values = [None,chest,float(chest_softmax),
    view,float(view_softmax[2]),float(view_softmax[0]),float(view_softmax[1]),df_id[df_id.columns[0]].values[0]]
    insert_query(table,columns,values)

    if view =='PA':
        df = opacity_detector(path)
        n_bb = len(df)
        max_conf = df['confidence'].max()
        print(max_conf)
        print(n_bb)
        p = -5.15022 + n_bb * 0.13448 + max_conf * 4.81532
        res = exp(p) / (1 + exp(p))
        print(res)
        if res > 0.75: 
            dict['Lung Opacity'] = df[['xmin','xmax','ymin','ymax','confidence']].to_dict(orient='index')
        else: 
            dict['Lung Opacity'] = 'Sin Hallazgos'
    return jsonify(dict)