def azureml_main(dataframe1=None, dataframe2=None):

    new_cols = pd.DataFrame(
        columns=["Algorithm"],
        data=[
            ["Boosted DT Regression"],
            ["Decision Forest Regression"],
            ["Linear Regression"]
        ]
    )
    result = pd.concat([new_cols, dataframe1], axis=1)
    return result,
	
	