import pandas as pd
import numpy as np
import os
import datetime

#### load climate data (e.g. precipitation, temperature, evapotranspiration)
def load_forcing(path,basin_cod,time=False):

    forcing_path= os.path.join(path,'CABra_climate_daily_series/climate_daily')

    forcing_df = pd.read_csv(os.path.join(forcing_path,'ref','CABra_'+str(basin_cod)+'_climate_'+'REF'+'.txt'),
                                       skiprows=13,encoding= 'unicode_escape',sep='\t')

    forcing_df.columns = forcing_df.columns.str.replace(' ', '')
    forcing_df=forcing_df.drop(0).astype('float32')
    dates = (forcing_df.Year.map(int).map(str) + "/" + forcing_df.Month.map(int).map(str) + "/"+  forcing_df.Day.map(int).map(str))
    forcing_df.index = pd.to_datetime(dates, format="%Y/%m/%d")
    #forcing_df=forcing_df.reset_index().drop("index",axis=1)
    forcing_df=forcing_df.fillna(forcing_df.mean(skipna=True))

    return forcing_df

### create column in table with time information
def load_forcing_time_information(df):
    df=df.reset_index()
    date_time = pd.to_datetime(df.pop('index'),  format="%Y/%m/%d")

    timestamp_s = date_time.map(datetime.datetime.timestamp)

    day = 24*60*60
    month=((30+31)/2)*day
    year = (366)*day


    df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
    df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return df


### load basin attributes
def basin_area(path,basin_cod):
    path_attributes=os.path.join(path,'CABra_attributes','CABra_topography_attributes.txt')
    data=pd.read_csv(path_attributes,skiprows=7,encoding= 'unicode_escape',sep='\t')
    data.columns = data.columns.str.replace(' ', '')
    data=data.drop(0).astype('float32')
    data=data.reset_index().drop("index",axis=1)
    area=data.loc[data['CABraID']==float(basin_cod)]['catch_area'].values
    return area

### laod discharge information
def load_discharge(path,basin_cod):

    attributes_cols=["Year",'Month','Day','Streamflow(m³s)','Quality']

    discharge_path =os.path.join(path,'CABra_streamflow_daily_series/streamflow_daily') #'/home/pedrozamboni/Documentos/doutorado/dataset/cabra/CABra_streamflow_daily_series/streamflow_daily'

    discharge= pd.read_csv(os.path.join(discharge_path,'CABra_'+str(basin_cod)+'_streamflow.txt'),

                                       skiprows=10,encoding= 'unicode_escape',sep='\t',names=attributes_cols)

    

    discharge.columns = discharge.columns.str.replace(' ', '')

    discharge=discharge.astype('float32')
    dates = (discharge.Year.map(int).map(str) + "/" + discharge.Month.map(int).map(str) + "/"+  discharge.Day.map(int).map(str))
    discharge.index = pd.to_datetime(dates, format="%Y/%m/%d")

    area=basin_area(path,basin_cod)
    discharge['streamflow(mm/dia)']= (discharge['Streamflow(m³s)']/(area*10**6))*86400*1000
    discharge=discharge.fillna(discharge.mean(skipna=True))

    return discharge

#### split the data into train, val and test
def train_test_split(df,train,val,test):
    n=len(df)
    train_df=df[0:int(n*train)]
    val_df=df[int(n*train):int(n*(train+val))]
    test_df=df[int(n*(train+val)):]
    
    return train_df,val_df,test_df

### create the sequence of data
def create_dataset(X, y, time_steps=1):

    Xs, ys = [], []

    for i in range(len(X) - time_steps):

        v = X.iloc[i:(i + time_steps)].values

        Xs.append(v)

        ys.append(y.iloc[i + time_steps])

    return np.array(Xs), np.array(ys)


### normalize the data
def data_normalizer(train_df,val_df,test_df):

    train_mean = train_df.mean()
    train_std = train_df.std()

    norm_train = (train_df - train_mean) / train_std
    norm_val = (val_df - train_mean) / train_std
    norm_test = (test_df - train_mean) / train_std

    return train_mean,train_std,norm_train,norm_val, norm_test

def calc_nse(obs: np.array, sim: np.array) -> float:
    
    # only consider time steps, where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator
    
    return nse_val

def kge(simulations, evaluation):
    """Original Kling-Gupta Efficiency (KGE) and its three components
    (r, α, β) as per `Gupta et al., 2009
    <https://doi.org/10.1016/j.jhydrol.2009.08.003>`_.
    Note, all four values KGE, r, α, β are returned, in this order.
    :Calculation Details:
        .. math::
           E_{\\text{KGE}} = 1 - \\sqrt{[r - 1]^2 + [\\alpha - 1]^2
           + [\\beta - 1]^2}
        .. math::
           r = \\frac{\\text{cov}(e, s)}{\\sigma({e}) \\cdot \\sigma(s)}
        .. math::
           \\alpha = \\frac{\\sigma(s)}{\\sigma(e)}
        .. math::
           \\beta = \\frac{\\mu(s)}{\\mu(e)}
        where *e* is the *evaluation* series, *s* is (one of) the
        *simulations* series, *cov* is the covariance, *σ* is the
        standard deviation, and *μ* is the arithmetic mean.
    """
    # calculate error in timing and dynamics r
    # (Pearson's correlation coefficient)
    sim_mean = np.mean(simulations, axis=0, dtype=np.float64)
    obs_mean = np.mean(evaluation, dtype=np.float64)

    r_num = np.sum((simulations - sim_mean) * (evaluation - obs_mean),
                   axis=0, dtype=np.float64)
    r_den = np.sqrt(np.sum((simulations - sim_mean) ** 2,
                           axis=0, dtype=np.float64)
                    * np.sum((evaluation - obs_mean) ** 2,
                             dtype=np.float64))
    r = r_num / r_den
    # calculate error in spread of flow alpha
    alpha = np.std(simulations, axis=0) / np.std(evaluation, dtype=np.float64)
    # calculate error in volume beta (bias of mean discharge)
    beta = (np.sum(simulations, axis=0, dtype=np.float64)
            / np.sum(evaluation, dtype=np.float64))
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return np.vstack((kge_, r, alpha, beta))

def get_basin_id(path):
    path=os.path.join(path,'CABra_climate_daily_series/climate_daily/ref')
    extension='txt'
    filenames = [f for f in os.listdir(path) if f.endswith(extension)]
    size=[]
    for name in filenames:
        size.append(os.path.getsize(os.path.join(path,name)) > 260)

    ids=[]
    for i in range(len(size)):
        if size[i]==True:
            pass
        else:
            a=filenames[i]
            ids.append(int(filenames[i].split('_')[1]))
        
    cabra_ids=[]
    for i in range(1,736):
        if i in ids:
            pass
        else:
            cabra_ids.append(i)
    return cabra_ids


def prep_data(path,basin_code,inicial,last,x_features,y_features,train_split=0.6,val_split=0.2,test_split=0.2,time_steps=120):
    ### load the data
    forcing=load_forcing(path,basin_code).loc[inicial:last]
    discharge=load_discharge(path,basin_code).loc[inicial:last]

	### split data
    forcing_train,forcing_val,forcing_test=train_test_split(forcing,train_split,
   val_split,test_split)
    discharge_train,discharge_val,discharge_test=train_test_split(discharge,train_split,
   val_split,test_split)

    ### normalizing the data
    x_mean,x_std,norm_forcing_train,norm_forcing_val, norm_forcing_test=data_normalizer(forcing_train,forcing_val
    ,forcing_test)

    y_mean,y_std,norm_discharge_train,norm_discharge_val, norm_discharge_test=data_normalizer(discharge_train,discharge_val
    ,discharge_test)

    ### create sequences of data
    x_train, y_train = create_dataset(forcing_train[x_features] ,discharge_train[y_features], time_steps)
    print(x_train.shape, y_train.shape)

    x_val, y_val = create_dataset(forcing_val[x_features], discharge_val[y_features], time_steps)
    print(x_val.shape, y_val.shape)

    x_test, y_test = create_dataset(forcing_test[x_features], discharge_test[y_features], time_steps)
    print(x_test.shape, y_test.shape)

    return x_train, y_train, x_val, y_val,x_test, y_test