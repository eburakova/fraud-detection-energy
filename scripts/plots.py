import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from plot import create_fraud_freq_pieplot, barplot_grouped_by_target 


################
### pie plot ### - as used for presentation 
################


def calculate_fraud_percentage(client_data): 
    """Calculate percentage fraud from the target col of the client_df.

    Args:
        client_data (pd.DataFrame): df with target col called target

    Returns:
        tuple: percentage for no fraud (0) and fraud (1)
    """
    counted_values = client_data.target.value_counts()
    total = sum(counted_values)
    percent_0 = counted_values[0] / total * 100
    percent_1 = counted_values[1] / total * 100
    return percent_0, percent_1

def create_fraud_freq_pieplot(client_data):
    """create a pretty pie plot that shows the frequency of fraud/ nofraud.

    Args:
        client_data (pd.DataFrame): needs to contain target variable
    """
    p0,p1= calculate_fraud_percentage(client_data)
    y = [p0, p1]
    pie_labels = ['no fraud', 'fraud']
    pie_pieces = [0, 0.3] # cut a piece out of the pie

    plt.pie(y, labels = pie_labels,
            explode = pie_pieces,
            autopct='%1.2f%%',  # format the and display percentages
            )
    plt.title('Frequency of gas and energy fraud in Tunisia')
    plt.show()



############################################
### simple barplots for data exploration ###  - for any categorical features grouped by target var
############################################

# import the function called 'barplot_grouped_by_target'
# it uses the 'aggregate_feature_by_target function' to calculate the needed metrics


def aggregate_feature_by_target(your_df, feature: str, target='target'):
    """ This function takes a categorical feature of your df and counts it grouped by the target variable.
        It also returns a df with the percentages.

    Args:
        your_df (pd.DataFrame): df that contains feature and target var for each client
        feature (str): name of feature to aggregate
        target (str, optional): name of target variable to group by. Defaults to 'target'.

    Returns:
        pd.DataFrame: grouped df
    """
    df_plot = your_df.groupby(feature, as_index=False)[target].value_counts()
    percent_col = []
    for i in df_plot[feature].unique():
        for t in df_plot[target].unique():
            a = sum(df_plot[(df_plot[feature] == i) & (df_plot[target] == t)]['count']) / sum(df_plot[df_plot[feature] == i]['count']) * 100
            percent_col.append(a)
    df_plot['percent'] = percent_col
    # TODO: sort values also in plot
    df_plot.sort_values([target, 'percent'], ascending=False, inplace=True)
    
    return df_plot


def barplot_grouped_by_target(your_df, feature: str, target='target', x_rotation=True):
    """This function takes the function 'aggregate_feature_by_target' to create a grouped df 
    and creates a simple barplot for the categorical feature. 

    Args:
        your_df (pd.DataFrame): df that contains feature and target var for each client
        feature (str): name of feature to plot
        target (str, optional): name of target variable to group by. Defaults to 'target'.
        x_rotation (bool, optional): Rotate x-axis labels for better readability. Defaults to True.

    Returns:
        pd.DataFrame: df with aggregated feature grouped by target (including percentage)
    """

    grouped_df = aggregate_feature_by_target(your_df, feature, target)
    sns.barplot(data=grouped_df, x=feature, y='count', hue=target)
    plt.title(f'Are there {feature}s with more frauds?')
    # TODO: increase fig size
    if x_rotation:
        plt.xticks(rotation=45)
    plt.show()
    return grouped_df