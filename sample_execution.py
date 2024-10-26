import os
import time
import new_search_gpu
import numpy as np
import pandas as pd
import data_provider.data_provider as data_provider
import torch

def sample_execution(create_df, fit_by_residual, device, to_disk):
    from tqdm import tqdm
    def create_feature_names(prefix, count):
        """ Helper function to create feature names for DataFrames. """
        return [f"{prefix}{i+1}" for i in range(count)]


    def create_dataframe(rows=1000, features=10000, join_key_domains=None, prefix='f'):
        if join_key_domains is None:
            join_key_domains = {'join_key': 1000}

        data = np.random.randint(low=0, high=100, size=(rows, features))
        feature_cols = [f'{prefix}{i+1}' for i in range(features)]

        df = pd.DataFrame(data, columns=feature_cols)

        for key, domain in join_key_domains.items():
            join_keys = np.random.choice(
                domain, size=rows, replace=True)
            df.insert(0, key, join_keys)

        return df, feature_cols

    # List to store paths of seller dataframes
    seller_dfs = []

    if create_df:
        buyer_df, buyer_features = create_dataframe(rows=10000, features=3, join_key_domains={'m': 100, 'n': 10}, prefix='b')
        buyer_df.to_csv("data/test_dataset/buyer/buyer_1.csv", index=False)
        print("buyer_df of 1000 rows and 10000 features has been created with join key being country and year")
        target_feature = buyer_features[0]  # 'b1'

        for i in tqdm(range(100), desc="Creating seller dataframes"):
            seller_df, features = create_dataframe(rows=10000, features=10000, join_key_domains={'m': 100, 'n': 10}, prefix=f's{i+1}_')
            seller_df.to_csv(f"data/test_dataset/seller/seller_{i+1}.csv", index=False)
            seller_dfs.append(f"data/test_dataset/seller/seller_{i+1}.csv")

    else:
        # When not creating dataframes, populate seller_dfs with existing csv files
        directory_path = "data/test_dataset/seller"
        all_files = os.listdir(directory_path)
        seller_dfs = [os.path.join(directory_path, f) for f in all_files if f.startswith("seller_") and f.endswith(".csv")]

    
    prepare_data = data_provider.PrepareBuyerSellers()

    buyer_features = create_feature_names('b', 3)
    if 'b1' in buyer_features:
        buyer_features.remove('b1')
    # Load buyer data
    buyer = data_provider.PrepareBuyer(data_path="data/test_dataset/buyer/buyer_1.csv", join_keys=[['m'], ['n']], one_target_feature=False,features=buyer_features, target_feature='b1', from_disk=True, need_to_clean_data=False)
    prepare_data.add_buyer(buyer)
    
    for i, seller_path in tqdm(enumerate(seller_dfs), total=len(seller_dfs), desc="Adding sellers"):

        seller_name = os.path.basename(seller_path).replace('.csv', '')
        seller_feature = create_feature_names(f's{i+1}_', 10000)
        seller = data_provider.PrepareSeller(data_path=seller_path, join_keys=[['m'], ['n']], features=seller_feature, from_disk=True, need_to_clean_data=False)
        prepare_data.add_seller(seller_name=seller_name, seller=seller)

    buyer_join_keys = list(prepare_data.get_buyer_join_keys())
    buyer_data = prepare_data.get_buyer_data()
    seller_data = prepare_data.get_seller_data()
    seller_names = prepare_data.get_sellers().get_seller_names()

    data_market = new_search_gpu.DataMarket(device=device)
    data_market.register_buyer(buyer_df=buyer_data, join_keys=buyer_join_keys, target_feature='b1', join_key_domains=prepare_data.get_domain(), fit_by_residual=fit_by_residual)

    for i, seller_name in tqdm(enumerate(seller_names), total=len(seller_names), desc="Registering sellers"):
        seller_df = seller_data[seller_name].get_data()
        join_keys = list(prepare_data.get_sellers().get_sellers()[seller_name].join_key_domains.keys())
        data_market.register_seller(seller_df=seller_df, join_keys=join_keys, seller_name=seller_name, join_key_domains=prepare_data.get_domain())
    s = time.time()
    search_engine = new_search_gpu.SearchEngine(data_market, fit_by_residual=fit_by_residual)
    augplan, augplan_acc, result_df = search_engine.start(iter=10)
    t = time.time()
    print(f"Time to search: {t-s}")
    augplan = search_engine.augplan
    augplan_acc = data_market.augplan_acc

    print("Augmented Plan Accuracy: ", augplan_acc)
    print("Augmented Plan: ", augplan)

    import matplotlib.pyplot as plt
    plt.plot(augplan_acc)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Augmented Plan Accuracy')
    plt.show()

    if to_disk:
        result_df.to_csv("output.csv", index=False)
        print("Result has been saved to output.csv")
    return augplan_acc, augplan, result_df

print("Scaled test started")
sample_execution(create_df=True, fit_by_residual=True, device='cuda', to_disk=True)
