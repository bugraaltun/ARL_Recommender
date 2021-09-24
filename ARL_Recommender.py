############################################
# ASSOCIATION RULE LEARNING
############################################

# Our aim is to suggest products to users in the product purchasing process
# by applying association rule learning to the online retail II dataset.

# 1. Data Preprocessing
# 2. Preparing ARL Data Structure (Invoice-Product Matrix)
# 3. Extracting Association Rules
# 4. Preparing the Script of the Study
# 5. Recommending a Product to Users at the Basket Stage

############################################
# Data Pre-Processing
############################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df.head()
df.describe().T

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C" , na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)
df.head()

##########################################
# Preparing ARL Data Structure
##########################################

df_ger = df[df["Country"] == "Germany"]
df_ger.head()
df_ger.shape

"""
We will create an Invoice-Product matrix, let's try to create the rows,
First we get the invoice and then groupby according to the describe. There are product names in the describe, and invoices in the invoice.
In this way, information will be displayed on which invoice contains how many of each product.
"""
df_ger.groupby(["Invoice","Description"]).agg({"Quantity" : "sum"}).head(10)


"""
We brought all the products in each invoice, 
but we want only one invoice code per line. 
We want the product names to be in the columns, and we just want to know if there is or not at the intersections.
"""
df_ger.groupby(["Invoice", "Description"]).agg({"Quantity" : "sum"}).unstack().iloc[:5, :5]


"""
Let's write 1 and 0 instead of quantity. When we said fillna, we said 0 to the places that wrote nan.
We want it to write 1 if product exists, or 0 if there is no product, for this we use applymap.
Applymap travels in all cells. Using apply map, we can browse in all cells. 
While browsing, it will print 1 if the cells it handles are greater than 0, otherwise it will print 0.
We are functionalizing it for convenience and if user makes id=true it will continue on stock id code, 
else it will continue through describe.
"""

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

inv_pro_df_ger = create_invoice_product_df(df_ger, id=True)
inv_pro_df_ger.head()

#Checking some products via their id's with check_id function

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_ger, 21987) # PACK OF 6 SKULL PAPER CUPS
check_id(df_ger, 23235) # STORAGE TIN VINTAGE LEAF
check_id(df_ger, 22747) # POPPY'S PLAYHOUSE BATHROOM

############################################
# Creating Association Rules
############################################

# First, the probabilities of all possible product associations are calculated.
frequent_itemsets = apriori(inv_pro_df_ger, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)

# Creating association rules
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()

# support: probability of seeing both
# confidence: probability of getting Y when X is taken.
# lift: When X is taken, the probability of getting Y increases by this much.

rules.sort_values("lift", ascending=False)

############################################
# Preparing the Script of the Study
############################################

import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

def create_rules(dataframe, id=True, country="Germany"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


df = retail_data_prep(df)
rules = create_rules(df)

############################################
# Making Product Suggestions to Users at the Basket Stage
############################################

# User 1 product id: 21987
# User 2 product id: 23235
# User 3 product id: 22747
product1 = check_id(df_ger, 21987) # PACK OF 6 SKULL PAPER CUPS
product2 = check_id(df_ger, 23235) # STORAGE TIN VINTAGE LEAF
product3 = check_id(df_ger, 22747) # POPPY'S PLAYHOUSE BATHROOM


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]


a = arl_recommender(rules, 21987, 3)
b = arl_recommender(rules, 23235, 3)
c = arl_recommender(rules, 22747, 3)

# RECOMMENDATION FOR ID=21987 (PACK OF 6 SKULL PAPER CUPS)

for i in a:
    print(check_id(df, i))

""" 

['SET OF 60 PANTRY DESIGN CAKE CASES ']

['RED RETROSPOT MINI CASES']

['REGENCY CAKESTAND 3 TIER']
"""
#RECOMMENDATION FOR ID=23235 (STORAGE TIN VINTAGE LEAF)

for i in b:
    print(check_id(df, i))

"""
['BLUE POLKADOT PLATE ']

['RED RETROSPOT MINI CASES']

['ROBOT BIRTHDAY CARD']

"""

#RECOMMENDATION FOR ID=22747 (POPPY'S PLAYHOUSE BATHROOM)

for i in c:
    print(check_id(df, i))

"""
['RED RETROSPOT MINI CASES']

['REGENCY CAKESTAND 3 TIER']

['PLASTERS IN TIN WOODLAND ANIMALS']

"""
