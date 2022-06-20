import os
import pandas as pd


class Olist:
    def get_data(self):
        """
        This function returns a Python dict.
        The keys are the file names and its values are pd.dataframes loaded
        the csv files.
        """
        csv_path = os.path.join(os.path.dirname(__file__), "../data/csv/")
        file_names = [
            "olist_sellers_dataset.csv",
            "product_category_name_translation.csv",
            "olist_orders_dataset.csv",
            "olist_order_items_dataset.csv",
            "olist_customers_dataset.csv",
            "olist_geolocation_dataset.csv",
            "olist_order_payments_dataset.csv",
            "olist_order_reviews_dataset.csv",
            "olist_products_dataset.csv",
        ]
        key_names = [i[6:-12] if i[:5] == "olist" else i[:-4] for i in file_names]
        data = {}
        for i, j in zip(key_names, file_names):
            data[i] = pd.read_csv(csv_path + j)
        return data

    def ping(self):
        """
        You call ping I print pong.
        """
        print("pong")


print(os.path.join(os.path.dirname(__file__), "../data/csv/"))
