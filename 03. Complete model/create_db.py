from queries import create_query, drop_query

images = {
    'id':'integer PRIMARY KEY AUTOINCREMENT NOT NULL',
    'path': 'text',
    'width':'integer NOT NULL',
    'height':'integer NOT NULL',
}

ssd_results = {
    'id':'integer PRIMARY KEY AUTOINCREMENT NOT NULL',
    'thorax':'integer NOT NULL',
    'thorax_softmax': 'real NOT NULL',
    'projection':'text',
    'pa_softmax':'real',
    'ap_softmax':'real',
    'l_softmax':'real',
    'bb_counts':'integer',
    'bb_cmax':'real',
    'lung_opacity':'integer',
    'output_image_path':'text',
    'image_id' : 'integer NOT NULL',
    'FOREIGN KEY(image_id)':'REFERENCES images (id)'
    } 


lung_opacity_detections = {
    'id':'integer PRIMARY KEY AUTOINCREMENT NOT NULL',
    'relative_x_center':'real NOT NULL',
    'relative_y_center':'real NOT NULL',
    'relative_width':'real NOT NULL',
    'relative_height':'real NOT NULL',
    'image_id':'integer NOT NULL',
    'FOREIGN KEY(image_id)':'REFERENCES images (id)'
    }

drop_query('images')
drop_query('ssd_results')
drop_query('lung_opacity_detections')
create_query('images',**images)
create_query('ssd_results',**ssd_results)
create_query('lung_opacity_detections',**lung_opacity_detections)