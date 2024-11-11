#Data formatting and processing
import json
import os
import numpy as np
import pandas as pd

def processData(data):
    for key in ['problem', 'algo']:
        if key in data:
            data[key] = ''.join(data[key])
    averageOnes = ["sp", "ps", "kt"]
    for key in data:
        if key in averageOnes:
            # Stack all 2D lists (matrices) into a 3D numpy array
            matrices = np.array(data[key])
            # Calculate the average across all matrices (along the first axis)
            average_matrix = np.mean(matrices, axis=0)
            # Replace the list of matrices with the average matrix
            data[key] = average_matrix.tolist()
    nextSet = ["cM", "s"]
    for key in data:
        if key in nextSet:
            aver = 0
            for val in data[key]:
                aver+=val
            data[key] = aver/len(data[key])
    return data

algos = ["NSGA2", "SPEA2",  "NSGA3", "MOEAD" , "SMSEMOA"]
problem = "DTLZ"
noOfProbs = 7

baseDirectory = os.getcwd()
AllData = []
#reacquire and process data
for i in range(1, noOfProbs+1):
    for algo in algos:
        try:
            filename = algo+"_"+problem+str(i)+"_2D.json"
            filePath = os.path.join(baseDirectory, algo, filename)
            with open(filePath, 'r') as file:
                data = json.load(file)
                data = processData(data)
                AllData.append(data)
        except:
            print("error in RD")


# Convert the list of dictionaries (AllData) to a pandas DataFrame
df = pd.DataFrame(AllData)

# Export the DataFrame to an Excel file
excel_file_path = os.path.join(baseDirectory, 'ProcessedData4DTLZ2D.xlsx')
df.to_excel(excel_file_path, index=False)

print(f"Data saved to {excel_file_path}")
#go through each problem instance and pick out some with a range of correlations. have table where all problems are accessed in appendices.

'''
TODO:
1. acquire data from each file
2. get average and extremeities from each field - 


put all data into a table - maybe excel file

'''