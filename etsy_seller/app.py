import os
from flask import render_template, request, Flask
from flask_pymongo import PyMongo
import pickle
import numpy as np
import pandas as pd

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(
        SECRET_KEY='dev',
        MONGO_URI='mongodb://localhost:27017/myDatabase',
    )
    
    kproto = pickle.load(open('kp_earring_20.pkl', 'rb'))
    xgb_reg = pickle.load(open('xgb_regressor.pkl', 'rb'))
    num_faves_scaler = pickle.load(open('num_faves_scaler.pkl', 'rb'))
    num_views_scaler = pickle.load(open('num_views_scaler.pkl', 'rb'))
    price_scaler = pickle.load(open('price_scaler.pkl', 'rb'))
    
    
    mongo = PyMongo(app)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    def cluster(**kwargs):
        # print(kwargs.keys())
        clust_label=0
        input_features = {}
        input_features['is_customizable'] = int(kwargs['customizable'])
        input_features['contains_silver'] = 0
        input_features['contains_gold'] = 0
        input_features['contains_glass'] = 0
        input_features['contains_diamond'] = 0
        input_features['contains_pearl'] = 0
        input_features['contains_stone'] = 0
        input_features['Chandelier Earrings'] = 0
        input_features['Clip On Earrings'] = 0
        input_features['Dangle & Drop Earrings'] = 0
        input_features['Ear Jackets & Climbers'] = 0
        input_features['Hoop Earrings'] = 0
        input_features['Stud Earrings'] = 0
        input_features['scaled_num_fav'] = 0
        input_features['scaled_views'] = 0
        input_features['scaled_price'] = 0
        input_features['is_single_item'] = 0
        input_features['when_made_cat'] = 0
        input_features['who_made_cat'] = 0


        if kwargs['material'] == 'Silver':
          input_features['contains_silver'] = 1
        elif kwargs['material'] == 'Gold':
          input_features['contains_gold'] = 1
        elif kwargs['material'] == 'Diamond':
          input_features['contains_diamond'] = 1
        elif kwargs['material'] == 'Pearl':
          input_features['contains_pearl'] = 1
        elif kwargs['material'] == 'Gemstone':
          input_features['contains_stone'] = 1
        elif kwargs['material'] == 'Glass':
          input_features['contains_glass'] = 1

        if kwargs['subcategory'] == 'Chandelier Earrings':
          input_features['Chandelier Earrings'] = 1
        elif kwargs['subcategory'] == 'Clip On Earrings':
          input_features['Clip On Earrings'] = 1
        elif kwargs['subcategory'] == 'Dangle & Drop Earrings':
          input_features['Dangle & Drop Earrings'] = 1
        elif kwargs['subcategory'] == 'Ear Jackets & Climbers':
          input_features['Ear Jackets & Climbers'] = 1
        elif kwargs['subcategory'] == 'Hoop Earrings':
          input_features['Hoop Earrings'] = 1
        elif kwargs['subcategory'] == 'Stud Earrings':
          input_features['Stud Earrings'] = 1

        # kwargs['category']
        
        input_features['is_single_item'] = 1 if kwargs['quantity'] == 1 else 0
        input_features['when_made_cat'] = 1 if kwargs['when'] == 'standard' \
                                         else 2 if kwargs['when'] == 'made_to_order' \
                                         else 3
        input_features['who_made_cat'] = 1 if kwargs['who'] == 'i_did' \
                                         else 2 if kwargs['who'] == 'someone_else' \
                                         else 3


        user_price = np.array(kwargs['user_price']).reshape(-1,1)
        scaled_views = np.array(kwargs['views']).reshape(-1,1)
        scaled_num_fav = np.array(kwargs['faves']).reshape(-1,1)
        input_features['scaled_price'] = price_scaler.transform(user_price)[0][0]
        input_features['scaled_views'] = num_views_scaler.transform(scaled_views)[0][0]
        input_features['scaled_num_fav'] = num_faves_scaler.transform(scaled_num_fav)[0][0]

        input_data = pd.DataFrame(input_features, index = [0])
        categorical_col = [0,1,2,3,4,5,6,7,8,9,10,11,12,16,17,18]
        try:
            pred = kproto.predict(input_data, categorical = categorical_col )
        except ValueError:
            print(input_data.columns)
            print(input_data.cluster_6)
            raise

        return pred[0]

    def predict_price(**kwargs):
        input_features = {}
        input_features['is_customizable'] = int(kwargs['customizable'])
        input_features['contains_silver'] = 0
        input_features['contains_gold'] = 0
        input_features['contains_glass'] = 0
        input_features['contains_diamond'] = 0
        input_features['contains_pearl'] = 0
        input_features['contains_stone'] = 0
        input_features['Chandelier Earrings'] = 0
        input_features['Clip On Earrings'] = 0
        input_features['Dangle & Drop Earrings'] = 0
        input_features['Ear Jackets & Climbers'] = 0
        input_features['Hoop Earrings'] = 0
        input_features['Stud Earrings'] = 0
        input_features['scaled_num_fav'] = 0
        input_features['scaled_views'] = 0
        input_features['is_single_item'] = 0
        input_features['when_made_cat'] = 0
        input_features['who_made_cat'] = 0


        if kwargs['material'] == 'Silver':
          input_features['contains_silver'] = 1
        elif kwargs['material'] == 'Gold':
          input_features['contains_gold'] = 1
        elif kwargs['material'] == 'Diamond':
          input_features['contains_diamond'] = 1
        elif kwargs['material'] == 'Pearl':
          input_features['contains_pearl'] = 1
        elif kwargs['material'] == 'Gemstone':
          input_features['contains_stone'] = 1
        elif kwargs['material'] == 'Glass':
          input_features['contains_glass'] = 1

        if kwargs['subcategory'] == 'Chandelier Earrings':
          input_features['Chandelier Earrings'] = 1
        elif kwargs['subcategory'] == 'Clip On Earrings':
          input_features['Clip On Earrings'] = 1
        elif kwargs['subcategory'] == 'Dangle & Drop Earrings':
          input_features['Dangle & Drop Earrings'] = 1
        elif kwargs['subcategory'] == 'Ear Jackets & Climbers':
          input_features['Ear Jackets & Climbers'] = 1
        elif kwargs['subcategory'] == 'Hoop Earrings':
          input_features['Hoop Earrings'] = 1
        elif kwargs['subcategory'] == 'Stud Earrings':
          input_features['Stud Earrings'] = 1
        
        input_features['is_single_item'] = 1 if kwargs['quantity'] == 1 else 0
        input_features['when_made_cat'] = 1 if kwargs['when'] == 'standard' \
                                         else 2 if kwargs['when'] == 'made_to_order' \
                                         else 3
        input_features['who_made_cat'] = 1 if kwargs['who'] == 'i_did' \
                                         else 2 if kwargs['who'] == 'someone_else' \
                                         else 3


        scaled_views = np.array(kwargs['views']).reshape(-1,1)
        scaled_num_fav = np.array(kwargs['faves']).reshape(-1,1)
        input_features['scaled_views'] = num_views_scaler.transform(scaled_views)[0][0]
        input_features['scaled_num_fav'] = num_faves_scaler.transform(scaled_num_fav)[0][0]

        when_key = str(kwargs['when'])
        input_features['made_to_order'] = 0
        input_features['vintage'] = 0
        input_features['standard'] = 0
        input_features[when_key] = 1

        who_key = str(kwargs['who'])
        input_features['someone_else'] = 0
        input_features['i_did'] = 0
        input_features['collective'] = 0
        input_features[who_key] = 1

        cluster_key = 'cluster_'+ str(kwargs['cluster_id'])
        input_features['cluster_0'] = 0
        input_features['cluster_1'] = 0
        input_features['cluster_2'] = 0
        input_features['cluster_3'] = 0
        input_features['cluster_4'] = 0
        input_features['cluster_5'] = 0
        input_features['cluster_6'] = 0
        input_features['cluster_7'] = 0
        input_features['cluster_8'] = 0
        input_features['cluster_9'] = 0
        input_features['cluster_10'] = 0
        input_features['cluster_11'] = 0
        input_features['cluster_12'] = 0
        input_features['cluster_13'] = 0
        input_features['cluster_14'] = 0
        input_features['cluster_15'] = 0
        input_features['cluster_16'] = 0
        input_features['cluster_17'] = 0
        input_features['cluster_18'] = 0
        input_features['cluster_19'] = 0
        input_features[cluster_key] = 1

        input_data = pd.DataFrame(input_features, index = [0])
        input_data = input_data.drop(columns = ['standard', 'i_did',
                                                'when_made_cat','who_made_cat'])

        return round(xgb_reg.predict(input_data)[0])

    @app.route('/', methods = ['POST', 'GET'])
    def hello():
        # category=None
        subcategory=None
        price=None
        customizable=None
        material=None
        who=None
        when=None
        quantity=None
        user_price=None
        views=None
        faves=None
        cluster_id=None
        if request.method == 'POST':
            # category = request.form['category']
            subcategory = request.form['subcategory']
            customizable = True if request.form['customizable']=='yes' else False
            material = request.form['material']
            who = request.form['who']
            when = request.form['when']
            quantity = int(request.form['quantity'])
            user_price = float(request.form['price'])
            views = int(request.form['views'])
            faves = int(request.form['faves'])

            # call model HERE
            cluster_id = cluster(subcategory=subcategory,
                                 customizable=customizable,
                                 material=material,
                                 who=who,
                                 when=when,
                                 quantity=quantity,
                                 user_price=user_price,
                                 views=views,
                                 faves=faves,
                                 )
            price = predict_price(cluster_id=cluster_id,
                                  subcategory=subcategory,
                                  customizable=customizable,
                                  material=material,
                                  who=who,
                                  when=when,
                                  quantity=quantity,
                                  views=views,
                                  faves=faves,
                                  )
        return render_template('index.html',
                                price=price,
                                subcategory=subcategory,
                                customizable=customizable,
                                material=material,
                                who=who,
                                when=when,
                                quantity=quantity,
                                user_price=user_price,
                                views=views,
                                faves=faves,
                                cluster_id=cluster_id,
                                )

    return app
