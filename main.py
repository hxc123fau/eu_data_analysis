import os
import pandas as pd
import numpy as np
from read_data import *
from factor_analyzer import FactorAnalyzer
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
from sklearn import metrics


class europe_data_analysis():

    def __init__(self, index_type):
        all_table_data = load_data().read_csv('./data')
        if index_type == 'HDI':
            self.table_data = \
                all_table_data[['country','median_income','life_expect']]
        elif index_type == 'HDI_more':
            self.table_data = \
                all_table_data[['country','gdp','life_expect','prct_rpt_crime',
                                'prct_budget_veryhard','legal_trust_rating','police_trust_rating',
                                'political_trust_rating',]]
        elif index_type == 'GGG':
            self.table_data = \
                all_table_data[['country','prct_low_savings','prct_yng_adt_pop',
                                'prct_budget_veryhard','prct_health_verygood',
                                'legal_trust_rating','political_trust_rating',
                                'police_trust_rating']]
        else:
            self.table_data = all_table_data

    def factor_analysis(self,input_x):
        ss_x = StandardScaler().fit_transform(input_x)
        norm_x = normalize(input_x,axis=0)
        factor_number = 9
        fa = FactorAnalyzer(n_factors=factor_number, rotation='oblimin')  # oblimin/promax varimax:orthogonal
        fa.fit(ss_x)
        ev, v = fa.get_eigenvalues()
        factor_loading_matrix = fa.loadings_
        fa_score = fa.transform(ss_x)
        print('ev',ev)
        # print('v',v)
        # print('factor_loading_matrix',factor_loading_matrix)
        fa_name = list(self.table_data.columns[1::])
        # print('quantization_score', len(fa_name),fa_name)
        for i in range(factor_number):
            all_coefficients = np.sort(factor_loading_matrix[:, i])
            coefficients_index = np.argsort(factor_loading_matrix[:, i])
            print('factor_i',i)
            for j,coefficient in enumerate (all_coefficients):
                if coefficient>0.5:
                    print('coefficients_index',coefficients_index[j],fa_name[coefficients_index[j]])


        plt.scatter(range(1, input_x.shape[1] + 1), ev)
        plt.plot(range(1, input_x.shape[1] + 1), ev)
        plt.title('scree figure')
        plt.ylabel('eigenvalues')
        plt.grid()
        plt.show()

        return fa_score

    def pca_analysis(self,input_x,component):
        # print('input_x',input_x.shape)
        ss_x = StandardScaler().fit_transform(input_x)
        norm_x = normalize(input_x,axis=1)
        pca = PCA()
        pca.fit(ss_x)
        # pca_res = pca.transform(ss_x)
        print('Explained Variance = ', pca.explained_variance_)
        sum_eigenvalues = np.sum(pca.explained_variance_)
        temp = 0
        eigenvalue_proportion = []
        for i in pca.explained_variance_:
            temp += i
            eigenvalue_proportion.append(temp / sum_eigenvalues)
        print('eigenvalue_proportion', eigenvalue_proportion)
        plt.grid()
        plt.title('pca_eigenvalue_proportion')
        # plt.scatter(eigenvalue_proportion)
        plt.plot(eigenvalue_proportion)
        plt.show()

        pca_2 = PCA(n_components=component)
        pca_2.fit(ss_x)
        pca_res = pca_2.transform(ss_x)
        # print('pca_res', pca_res.shape)

        return pca_res

    def lle_analysis(self,input_x):
        ss_x = StandardScaler().fit_transform(input_x)
        norm_x = normalize(input_x, axis=0)
        n_neighbors = 5
        # LLE降维
        n_components = 4
        lle_res = manifold.LocallyLinearEmbedding(n_neighbors, n_components).fit_transform(ss_x)
        # print('lle_res',lle_res)

        return lle_res

    def multivariate_analysis(self, input_x, y_label):
        # x = StandardScaler().fit_transform(input_x)
        # y=StandardScaler().fit_transform(y_label)
        all_predict=[]
        all_label=[]
        all_coef_=[]
        for i in range(5):
            x_train, x_test, y_train, y_test = train_test_split(input_x, y_label, test_size=0.2)
            lr_model = LinearRegression()
            lr_model.fit(x_train, y_train)
            predict_data = lr_model.predict(x_test)
            all_predict.extend(predict_data)
            all_label.extend(y_test)
            all_coef_.append(lr_model.coef_[0])
        plt.plot(all_predict,c='r')
        plt.plot(all_label, c='b')
        plt.title('multivariate analysis')
        plt.legend(["predict", "label"], loc="upper right")
        plt.show()
        all_coef_np=np.array(all_coef_)
        average_coef_np=np.average(all_coef_np,axis=0)
        print('average_coef_np',average_coef_np.shape)

        return average_coef_np

    def k_means_clustering(self, input_x):
        # print('input_x',input_x)
        scale_x = StandardScaler().fit_transform(input_x)
        # print('scale_x',scale_x)
        norm_x = normalize(input_x,axis=1)
        # print('norm_x',norm_x)
        k_means_model = KMeans(n_clusters=2, random_state=0).fit(norm_x)
        k_means_res = k_means_model.labels_

        return k_means_res

    def spectral_clustering(self,input_x):
        ss_x = StandardScaler().fit_transform(input_x)
        norm_x = normalize(input_x)
        spectral_res = []
        # y_pred = SpectralClustering().fit_predict(input_x)
        for index, gamma in enumerate((1e-04, 1e-03, 0.01, 0.1, 1, 10)):
            y_pred = SpectralClustering(n_clusters=2, gamma=gamma).fit_predict(ss_x)
            # print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=",
            #           2, "score:", metrics.calinski_harabasz_score(ss_x, y_pred))
            # print('y_pred',y_pred)
            spectral_res.append(y_pred)
        y_pred_final = SpectralClustering(n_clusters=2, gamma=0.1).fit_predict(ss_x)

        return y_pred_final


    def gini_index(self,income_proportion):
        # income_proportion = [12, 15.4, 36.7, 25.4, 7.7, 2.7]
        ideal_income = [np.power(2,1),np.power(2,3), np.power(2,4),
                        np.power(2,5), np.power(2,8), np.power(2,10)]
        gdp_percent = np.multiply(ideal_income,income_proportion)
        total = np.cumsum(gdp_percent)
        gdp_decile_percents = gdp_percent / total[-1] * 100.0
        # print('Percents sum to 100:', sum(gdp_decile_percents) == 100)
        gdp_decile_shares = [i / 100 for i in gdp_decile_percents]
        # Convert to quintile shares of total GDP
        gdp_quintile_shares = [(gdp_decile_shares[i]) for i in range(0, len(gdp_decile_shares), 1)]
        gdp_quintile_shares.insert(0,0)
        # Cumulative sum of shares (Lorenz curve values)
        shares_cumsum = np.cumsum(a=gdp_quintile_shares, axis=None)

        # Perfect equality line
        pe_line = np.linspace(start=0.0, stop=1.0, num=len(shares_cumsum))
        area_under_lorenz = np.trapz(y=shares_cumsum, dx=1 / len(shares_cumsum))
        area_under_pe = np.trapz(y=pe_line, dx=1 / len(shares_cumsum))
        gini = (area_under_pe - area_under_lorenz) / area_under_pe

        return shares_cumsum,pe_line,gini


def md_factor_analysis():
    mv_model = europe_data_analysis('all')
    input_vec = mv_model.table_data.to_numpy()[:, 1:]
    country_name = mv_model.table_data['country']

    fa_score = mv_model.factor_analysis(input_vec)
    # km_fa_res = mv_model.k_means_clustering(fa_score)

# Domestic saving/investment, Demographic prospects, Health, Education,
# Quality of institutions and policies, Trade openness
def lle_pca_vec_clustering():
    mv_model = europe_data_analysis('GGG')
    input_vec = mv_model.table_data.to_numpy()[:, 1:]
    country_name = mv_model.table_data['country']

    # lle_vec = mv_model.lle_analysis(input_vec)
    # km_res = mv_model.k_means_clustering(lle_vec)
    pca_vec = mv_model.pca_analysis(input_vec,4)
    km_res = mv_model.k_means_clustering(pca_vec)

    km_res = np.expand_dims(km_res, axis=1)
    data_clustering = np.concatenate((km_res, input_vec), axis=1)
    for i in range(data_clustering.shape[0]):
        if data_clustering[i, 0] == 0:
            plt.scatter(data_clustering[i, 1], data_clustering[i, 2], c='r')
            plt.annotate(mv_model.table_data.to_numpy()[:, 0][i], xy=(data_clustering[i, 1], data_clustering[i, 2]))
        else:
            plt.scatter(data_clustering[i, 1], data_clustering[i, 2], c='b')
            plt.annotate(mv_model.table_data.to_numpy()[:, 0][i], xy=(data_clustering[i, 1], data_clustering[i, 2]))
    plt.title('Global Growth Generating')
    plt.show()


def HDI_kmean_clustering():
    mv_model = europe_data_analysis('HDI')
    input_vec = mv_model.table_data.to_numpy()[:, 1:]
    country_name = mv_model.table_data['country']

    km_res = mv_model.k_means_clustering(input_vec)
    km_res = np.expand_dims(km_res, axis=1)
    data_clustering = np.concatenate((km_res, input_vec), axis=1)
    for i in range(data_clustering.shape[0]):
        if data_clustering[i, 0] == 0:
            plt.scatter(data_clustering[i, 1], data_clustering[i, 2], c='r')
            plt.annotate(mv_model.table_data.to_numpy()[:, 0][i], xy=(data_clustering[i, 1], data_clustering[i, 2]))
        else:
            plt.scatter(data_clustering[i, 1], data_clustering[i, 2], c='b')
            plt.annotate(mv_model.table_data.to_numpy()[:, 0][i], xy=(data_clustering[i, 1], data_clustering[i, 2]))
    plt.title('HDI_k_means')
    plt.show()
    # for i in zip(country_name, km_res):
    #         print('iii', i)


# crime rate multivarite analysis
def crime_rate_multivarite_analysis():
    mv_model = europe_data_analysis('all')
    country_name = mv_model.table_data['country']

    crime_input = mv_model.table_data[['prct_rpt_crime']].to_numpy()
    multi_input_column = mv_model.table_data[mv_model.table_data.columns.difference(['prct_rpt_crime', 'country'])]
    multi_var_input = multi_input_column[multi_input_column.columns[1:]].to_numpy()
    lr_coefficient = mv_model.multivariate_analysis(multi_var_input, crime_input)
    list_lr_coefficient = list(lr_coefficient)
    for i, coefficient in enumerate(list_lr_coefficient):
        var_name = multi_input_column.columns[i + 1]
        print('variable_importance', var_name, coefficient)

def gini_calculation():
    mv_model = europe_data_analysis('all')
    input_vec = pd.read_csv('./data/europe_dataset_make_ends_meet_2016.csv')
    # print('input_vec',input_vec.shape,input_vec)
    income_grade=input_vec.iloc[:,1:].to_numpy()
    country=input_vec.iloc[:,0]
    print('income_grade',income_grade.shape,type(income_grade))
    for i in range(32):
        # income_grade[i]
        wealth_cumulate,pe_line,gini=mv_model.gini_index(income_grade[i])
        order=i+1
        # plt.subplot(4,8, order)
        plt.plot(wealth_cumulate)
        plt.plot(pe_line)
        plt.title(country[i]+' gini ='+str("%.3f" % gini))

        plt.show()


if __name__ == "__main__":
    # HDI_kmean_clustering()
    md_factor_analysis()
    # lle_pca_vec_clustering()
    # crime_rate_multivarite_analysis()
    # gini_calculation()
