import warnings
warnings.filterwarnings('ignore')

import json, os, re, scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as mcolors
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions



# This finds our json files
print("This finds our json files")
loop = True
while loop:
#     path_to_json = str(input("Directory of Json files (which should starts and ends with /): "))
    path_to_json = '/Users/alivarastehranjbar/Documents/untitledfolder/Python/TNT-Lab/CTN_results/'
    # Store the in List called: json_files
    try:
        # Store the in List called: json_files
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    except:
        # if path is empty will get Error
        print("Error")

    if len(json_files) != 0:
        loop = False

# for me: '/Users/alivarastehranjbar/Documents/untitledfolder/Python/TNT-Lab/CTN_results/'
#/Users/alivarastehranjbar/Documents/untitledfolder/Python/TNT-Lab/CTN_results/



# Here I define a list to store each json file as a DataFrame in a list
json_list = list()

# we need both the json and an index number so use enumerate()
for index,js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        # Finding the location of gNodeB
        g_loc = re.findall('[x-y-p][\d]{1,100}',js)

        #  Loading Json
        json_text = json.load(json_file,parse_float=True)

        # Create DataFrame
        j_pd = pd.DataFrame.from_dict(json_text.items(),dtype=float)
        j_pd.columns = ['Location','BLER'+str(' in ')+str(g_loc[0])+str('-')+str(g_loc[1])+str('-')+str(g_loc[2])]

        # For taking locations as a seperate DF
        if index == 0:
            # Append seperatly
            json_list.append(j_pd[['Location']])
            json_list.append(j_pd[['BLER'+str(' in ')+str(g_loc[0])+str('-')+str(g_loc[1])+str('-')+str(g_loc[2])]])
        else :
            # for rest of indexes except '0'
            json_list.append(j_pd[['BLER'+str(' in ')+str(g_loc[0])+str('-')+str(g_loc[1])+str('-')+str(g_loc[2])]])

# Example of js
# bs_uc3_ls50_ws50_x75_y50_n5000_p100.json


# In this part we make our final DataFrame

# combine DataFrames Except Location
final = json_list[1]
for i in range(2,len(json_list)):
    final = pd.concat([final, json_list[i]], axis=1)

#-----------------------------------------------
#  human sorting (also known as natural sorting):
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
#-----------------------------------------------

# sort the columns by using human sorting
final = final.reindex(sorted(final.columns, key=natural_keys), axis=1)

# Add Location DataFrame to the final DataFrame
final = pd.concat([json_list[0], final], axis=1)

# Convert Location of UE to Numpy Array
UE_Loc = np.zeros([2500,2])
for i in range(len(final['Location'])):
    d = re.findall('[\d]{0,100}',final['Location'][i])
    UE_Loc[i] = [int(d[0])/2,int(d[4])/2]

# Create DataFrame of the UE Location
locdf=pd.DataFrame(UE_Loc,columns=['X','Y'])

# Add locatio DataFrame to the Main DataFrame
final = pd.concat([locdf, final], axis=1)
del final['Location']

# Convert Location of gNodeB to Numpy Array
gNodeB_Loc = np.zeros([int((len(final.columns)-2)/4),2])
for i in range(0,len(gNodeB_Loc)):
    d = re.findall('[\d]{2,3}',final.columns[i*4+2])
    gNodeB_Loc[i] = [int(d[0])/2,int(d[1])/2]

print(final.head())
print(final[final.columns[2:]].describe())

## Defining the Colors for ploting
# Colors which will be used in plots
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:pink', 'tab:olive', 'tab:gray', 'tab:brown', 'tab:orange', 'tab:purple']

sort_colors = True
if sort_colors is True:
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),name) for name, color in mcolors.CSS4_COLORS.items())
    names = [name for hsv, name in by_hsv]
else:
    names = list(colors)




##------------------------------------------------------------------------------
# Finding the best Fit for DataFrame

# 3D contour plot lines
numberOfContourLines = 16
graphWidth = 800 # units are pixels
graphHeight = 600 # units are pixels


## Class for finding the fit of the data and ploting the results
class fit_func:
    '''
    this class in put is the Data and the A0
    Dataset name
    Data is the x,y,z
    A0 is the location of the gNodeB of the data which is ploting
    '''
    def __init__(self,dataset_name,data,A0):
        self.dataset_name = dataset_name
        self.data = data
        self.A0 = A0

    def dist_plot(self, dataset_name):
        '''
        this def will get the data and plots the distribution and
        find the best fit for the data distribution to get the sigma and
        the mean of the data
        '''
        # sns.set_style('white')
        # sns.set_context("paper", font_scale = 2)
        # sns.displot(data=final[[dataset_name]], kind="hist", bins = 1000, aspect = 1.5)

        BLER = final[[dataset_name]].values
        f = Fitter(BLER,distributions=['gamma','lognorm',"beta","burr","norm"])

        f.fit()
        print(f.summary())
        best_fit = f.get_best(method = 'sumsquare_error')
        # key = best_fit.keys()
        # key, value = best_fit.items()
        if best_fit.keys() == 'lognorm':
            for key, value in best_fit.items():
                print("best fit is {}".format(key))
                for k, v in value.items():
                    if k == 's':
                        s = v
                    if k == 'loc':
                        loc = v
                    if k == 'scale':
                        scale = v
            sigma = s
            mean_1 = scipy.stats.lognorm.mean(s, loc=loc, scale=scale)
            print(best_fit, sigma, mean_1)

    def ScatterPlot(self,data):
        '''
        this def will plots the scatter plot of the Data
        '''
        f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)

        plt.grid(True)
        axes = Axes3D(f)
        x_data = data[0]
        y_data = data[1]
        z_data = data[2]

        axes.scatter(x_data, y_data, z_data)
        axes.set_title('Scatter Plot for {} Data'.format(2))
        axes.set_xlabel('X Data')
        axes.set_ylabel('Y Data')
        axes.set_zlabel('BLER Data')

        plt.show()
        plt.close('all') # clean up after using pyplot or else thaere can be memory and process problems



    def SurfacePlot(self,func, data,fittedParameters):
        '''
        this def will plots the data and the surface fit
        '''
        f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
        plt.grid(True)
        axes = Axes3D(f)

        # Data of the plot
        x_data = data[0]
        y_data = data[1]
        z_data = data[2]

        # defining the X and Y amd the Z of plot
        xModel = np.linspace(min(x_data), max(x_data), int(np.sqrt(len(x_data))))
        yModel = np.linspace(min(y_data), max(y_data), int(np.sqrt(len(y_data))))
        X, Y = np.meshgrid(xModel, yModel)
        # Z = func(numpy.array([X, Y]), *fittedParameters)
        Z = self.func([data[0], data[1]], *fittedParameters)
        Z = Z.reshape(len(xModel),len(yModel))

        axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=True)

        axes.scatter(x_data, y_data, z_data) # show data along with plotted surface

        axes.set_title('Surface Plot for {} Data and the Fit Surface'.format(2)) # add a title for surface plot
        axes.set_xlabel('X Data') # X axis data label
        axes.set_ylabel('Y Data') # Y axis data label
        axes.set_zlabel('BLER') # Z axis data label

        plt.show()
        plt.close('all') # clean up after using pyplot or else thaere can be memory and process problems

    def ContourPlot(self,func, data, fittedParameters):
        '''
        this def will plot the counter plot of the data to have
        the sight of BLER zones
        '''
        f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
        axes = f.add_subplot(111)

        x_data = data[0]
        y_data = data[1]
        z_data = data[2]

        xModel = np.linspace(min(x_data), max(x_data), int(np.sqrt(len(x_data))))
        yModel = np.linspace(min(y_data), max(y_data), int(np.sqrt(len(y_data))))
        X, Y = np.meshgrid(xModel, yModel)

        # Z = func(numpy.array([X, Y]), *fittedParameters)
        Z = self.func([data[0], data[1]], *fittedParameters)
        Z = Z.reshape(len(xModel),len(yModel))

        axes.plot(x_data, y_data, 'o')

        axes.set_title('Contour Plot for {} Data and shows the zone of BLERs'.format(2)) # add a title for contour plot
        axes.set_xlabel('X Data') # X axis data label
        axes.set_ylabel('Y Data') # Y axis data label

        CS = plt.contour(X, Y, Z, numberOfContourLines, colors='k')
        plt.clabel(CS, inline=1, fontsize=10) # labels for contours

        plt.show()
        plt.close('all') # clean up after using pyplot or else thaere can be memory and process problems

    def func(self,data, alpha, beta,a ,b):
        x1 = data[0]
        y1 = data[1]
        # A0 = data[2]
        ## 2D
        x = np.linspace(min(x1), max(x1), int(np.sqrt(len(x1))))
        y = np.linspace(min(y1), max(y1), int(np.sqrt(len(y1))))
        X , Y = np.meshgrid(x,y)
        Z = alpha * (((X-A0[0]+a)**2)) + beta*((Y-A0[1]+b)**2)
        return Z.ravel()


    def fit_c(self):
        '''
        this the is using ...
        for fit...
        boundaries ...
        method ...
        maxfev ...
        x_scale and f f_scale ...
        telorance ...
        loss ...
        function ...
        '''
        x,y = self.data[0],self.data[1]

        # defining fitting Function

        loop = True
        while loop:

            # getting mean and sigma of data
            self.dist_plot(self.dataset_name)

            initialParameters = [.0001, .0001, 0 , 6]
            # here a non-linear surface fit is made with scipy's curve_fit()
            tol = 10**-15
            fittedParameters, pcov = scipy.optimize.curve_fit(self.func, [x,y], z, bounds=([ 0, 0, -10, -10], [ .001, .001, 10, 10]),method='trf',
                                                                p0 = initialParameters,maxfev=10000,ftol=tol, xtol=tol, gtol=tol,
                                                                x_scale=0.1, loss='cauchy', f_scale=0.1, diff_step=None, verbose = 2)
            # ,sigma = 2.520656362227073/zData,absolute_sigma=False,maxfev=1000
            # sigma has been taken from fitter library and the fit was lognoraml
            # scipy.optimize.minimize()

            print('fitted prameters', fittedParameters)
            modelPredictions = self.func(data, *fittedParameters)
            absError = modelPredictions - z

            SE = np.square(absError) # squared errors
            MSE = np.mean(SE) # mean squared errors
            RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
            Rsquared = 1.0 - (np.var(absError) / np.var(z))

            # Condition for stopping the loop
            if (RMSE == RMSE and Rsquared == Rsquared):
                loop = False
                # ploting
                # ScatterPlot(data)
                print('RMSE:', RMSE)
                print('R-squared:', Rsquared)
                self.SurfacePlot(self.func, data,fittedParameters)
                self.ContourPlot(self.func, data, fittedParameters)
                return Rsquared,RMSE,fittedParameters


if __name__ == "__main__":
    # Defining Data
    x = np.array(final['X'])
    y = np.array(final['Y'])
    RR_list = list()

    # this loops will only consider the p = 50
    for i in range(2,len(final.columns),4):
        # A0 is the location of the gNodeB for the exact data
        A0 = gNodeB_Loc[int(abs(i/4)),:]

        print("\n calculating fit for gNodeB location: {} and data: {} ".format(A0,final.columns[i]))

        z = np.array(final[final.columns[i]])
        data = [x, y, z]


        # defining model
        model = fit_func(final.columns[i],data,A0)

        # Taking the RMSE and the Rsquared
        Rsquared,RMSE,fittedParameters = model.fit_c()

        # saving data in the RR_list
        RR_list.append([final.columns[i],Rsquared,RMSE,fittedParameters])
