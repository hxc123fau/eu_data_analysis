import os
import pandas as pd
import numpy as np



class load_data():
    def __init__(self):
        pass

    def read_csv(self,path):
        direct_list=os.listdir(path)
        data_list=[]
        # print('datacountry',data['country'])
        # print('new_data_frame',new_data_frame)
        all_column_headers=[]
        for i,file_name in enumerate (direct_list):
            read_file_name='./data/'+file_name
            data=pd.read_csv(read_file_name)
            column_headers = list(data.columns.values)
            # print('column_headers',column_headers)
            for column_name in column_headers:
                if 'Unnamed' in column_name:
                    data = data.drop(column_name, axis=1)

            all_column_headers.extend(column_headers)

            if 'europe_dataset_gdp_2016.csv' not in read_file_name \
                    and'europe_dataset_population_2011.csv' not in read_file_name \
                    and'europe_dataset_weather.csv' not in read_file_name:
                data_np = data.to_numpy()
                data_np_new = data_np[:, 1:]
                score = np.ones_like(data_np_new)
                column_number = data_np_new.shape[1]
                if file_name == 'europe_dataset_make_ends_meet_2016.csv':
                    for j in range(column_number):
                        score[:, j] = 10.0 / column_number * (1 + j) * data_np_new[:, j]
                elif column_number>1:
                    for j in range(column_number):
                        score[:, j] = 10.0 / column_number * (column_number - j) * data_np_new[:, j]
                else:
                    score=data_np_new

                average_data = np.average(score, axis=1)
                # data_country=np.expand_dims(data_np[:,0],axis=1)
                data_country=data_np[:,0]
                # average_data=np.expand_dims(average_data,axis=1)
                data_table = pd.DataFrame(average_data, columns=[column_headers[1]],index=data_country)
                data_list.append(data_table)

            if file_name == 'europe_dataset_gdp_2016.csv':  # polulation
                gdp_data_np = data.to_numpy()
                gdp_data = gdp_data_np[:, 1]
                gdp_country = gdp_data_np[:, 0]
                gdp_data_table = pd.DataFrame(gdp_data, columns=[column_headers[1]], index=gdp_country)

                data2 = pd.read_csv('./data/europe_dataset_population_2011.csv')
                population_data_np = data2.to_numpy()
                population_data = population_data_np[:, 1]
                population_country=population_data_np[:, 0]
                population_data_table = pd.DataFrame(population_data, columns=[column_headers[1]], index=population_country)

                # print('len(population_data_table)',len(population_data_table))
                country_name=[]
                gdp_capita_value=[]
                for i in range(len(gdp_data_table)):
                    for j in range(len(population_data_table)):
                        # print('iii',i,j,gdp_data_table.index[i],population_data_table.index[j])
                        if gdp_data_table.index[i] == population_data_table.index[j]:
                            # print('iii',i,j,gdp_data_table.index[i],population_data_table.index[j])
                            # print('111',gdp_data_table.iloc[i])
                            # print('222',population_data_table.iloc[j])
                            res=gdp_data_table.iloc[i]/population_data_table.iloc[j]
                            # print('res',res)
                            country_name.append(gdp_data_table.index[i])
                            gdp_capita_value.append(res)
                gdp_capita=pd.DataFrame(gdp_capita_value,columns=['gdp'],index=country_name)
                # print('gdp_capita',gdp_capita)
                data_list.append(gdp_capita)


            if file_name == 'europe_dataset_population_2011.csv':
                data_np = data.to_numpy()
                data_population = data_np[:, 2]
                data_country=data_np[:, 0]
                data_table = pd.DataFrame(data_population, columns=[column_headers[2]],index=data_country)
                data_list.append(data_table)

            if file_name == 'europe_dataset_weather.csv':
                data_np = data.to_numpy()
                data_avg_temp = data_np[:, 1]
                data_country=data_np[:, 0]
                data_table = pd.DataFrame(data_avg_temp, columns=[column_headers[1]],index=data_country)
                data_list.append(data_table)

                data_avg_precipitation = data_np[:, 4]
                data_table2 = pd.DataFrame(data_avg_precipitation, columns=[column_headers[4]],index=data_country)
                data_list.append(data_table2)


        first_data_frame=data_list[0]
        for data_f in data_list[1:]:
            # print('data_f',data_f)
            first_data_frame=first_data_frame.join(data_f)
            # print('first_data_frame',first_data_frame)

        # country_name=first_data_frame.ind
        # first_data_frame.insert(data.shape[1], 'd', 0)
        # print('first_data_frame',first_data_frame.index)
        first_data_frame.insert(loc=0,column='country',value=first_data_frame.index)
        first_data_frame.to_csv('./res.csv', index=False)
        # print('all_csv',first_data_frame)

        return first_data_frame

# load_data().read_csv('./data')