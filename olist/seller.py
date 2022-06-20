import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order


class Seller:
    def __init__(self):
        # Import data
        olist = Olist()
        self.data = olist.get_data()
        self.order = Order()

    def get_seller_features(self):
        """
        Returns a DataFrame with:
        'seller_id', 'seller_city', 'seller_state'
        """
        sellers = self.data["sellers"].copy()
        sellers.drop("seller_zip_code_prefix", axis=1, inplace=True)
        sellers.drop_duplicates(inplace=True)  # There can be multiple rows per seller
        return sellers

    def get_seller_delay_wait_time(self):
        """
        Returns a DataFrame with:
        'seller_id', 'delay_to_carrier', 'wait_time'
        """
        # Get data
        order_items = self.data["order_items"].copy()
        orders = self.data["orders"].query("order_status=='delivered'").copy()

        ship = order_items.merge(orders, on="order_id")

        # Handle datetime
        ship.loc[:, "shipping_limit_date"] = pd.to_datetime(ship["shipping_limit_date"])
        ship.loc[:, "order_delivered_carrier_date"] = pd.to_datetime(
            ship["order_delivered_carrier_date"]
        )
        ship.loc[:, "order_delivered_customer_date"] = pd.to_datetime(
            ship["order_delivered_customer_date"]
        )
        ship.loc[:, "order_purchase_timestamp"] = pd.to_datetime(
            ship["order_purchase_timestamp"]
        )

        # Compute delay and wait_time
        def delay_to_logistic_partner(d):
            days = np.mean(
                (d.order_delivered_carrier_date - d.shipping_limit_date)
                / np.timedelta64(24, "h")
            )
            if days > 0:
                return days
            else:
                return 0

        def order_wait_time(d):
            days = np.mean(
                (d.order_delivered_customer_date - d.order_purchase_timestamp)
                / np.timedelta64(24, "h")
            )
            return days

        delay = ship.groupby("seller_id").apply(delay_to_logistic_partner).reset_index()
        delay.columns = ["seller_id", "delay_to_carrier"]

        wait = ship.groupby("seller_id").apply(order_wait_time).reset_index()
        wait.columns = ["seller_id", "wait_time"]

        df = delay.merge(wait, on="seller_id")

        return df

    def get_active_dates(self):
        """
        Returns a DataFrame with:
        'seller_id', 'date_first_sale', 'date_last_sale', 'months_on_olist'
        """
        # First, get only orders that are approved
        orders_approved = self.data["orders"][
            ["order_id", "order_approved_at"]
        ].dropna()

        # Then, create a (orders <> sellers) join table because a seller can appear multiple times in the same order
        orders_sellers = orders_approved.merge(self.data["order_items"], on="order_id")[
            ["order_id", "seller_id", "order_approved_at"]
        ].drop_duplicates()
        orders_sellers["order_approved_at"] = pd.to_datetime(
            orders_sellers["order_approved_at"]
        )

        # Compute dates
        orders_sellers["date_first_sale"] = orders_sellers["order_approved_at"]
        orders_sellers["date_last_sale"] = orders_sellers["order_approved_at"]
        df = orders_sellers.groupby("seller_id").agg(
            {"date_first_sale": min, "date_last_sale": max}
        )
        df["months_on_olist"] = round(
            (df["date_last_sale"] - df["date_first_sale"]) / np.timedelta64(1, "M")
        )
        return df

    def get_quantity(self):
        """
        Returns a DataFrame with:
        'seller_id', 'n_orders', 'quantity', 'quantity_per_order'
        """
        order_items = self.data["order_items"]

        n_orders = order_items.groupby("seller_id")["order_id"].nunique().reset_index()
        n_orders.columns = ["seller_id", "n_orders"]

        quantity = order_items.groupby("seller_id", as_index=False).agg(
            {"order_id": "count"}
        )
        quantity.columns = ["seller_id", "quantity"]

        result = n_orders.merge(quantity, on="seller_id")
        result["quantity_per_order"] = result["quantity"] / result["n_orders"]
        return result

    def get_sales(self):
        """
        Returns a DataFrame with:
        'seller_id', 'sales'
        """
        return (
            self.data["order_items"][["seller_id", "price"]]
            .groupby("seller_id")
            .sum()
            .rename(columns={"price": "sales"})
        )

    def get_review_score(self):
        """
        Returns a DataFrame with:
        'seller_id', 'share_of_five_stars', 'share_of_one_stars', 'review_score'
        """
        seller_reviews = pd.merge(
            self.data["sellers"][["seller_id"]].merge(
                self.data["order_items"][["order_id", "seller_id"]],
                on="seller_id",
                how="outer",
            ),
            self.data["order_reviews"][["review_id", "order_id", "review_score"]],
            on="order_id",
            how="outer",
        )
        seller_reviews = seller_reviews[["seller_id", "review_id", "review_score"]]
        seller_reviews["share_of_five_stars"] = seller_reviews["review_score"].map(
            lambda x: 1 if x == 5 else 0
        )
        seller_reviews["share_of_one_stars"] = seller_reviews["review_score"].map(
            lambda x: 1 if x == 1 else 0
        )
        lam_1 = lambda x: x.sum() / x.count()
        outp = seller_reviews.groupby("seller_id").agg(
            {
                "review_score": np.mean,
                "share_of_five_stars": lam_1,
                "share_of_one_stars": lam_1,
            }
        )
        return outp.dropna()

    def get_training_data(self):
        """
        Returns a DataFrame with:
        ['seller_id', 'seller_city', 'seller_state', 'delay_to_carrier',
        'wait_time', 'date_first_sale', 'date_last_sale', 'months_on_olist', 'share_of_one_stars',
        'share_of_five_stars', 'review_score', 'n_orders', 'quantity',
        'quantity_per_order', 'sales']
        """

        training_set = (
            self.get_seller_features()
            .merge(self.get_seller_delay_wait_time(), on="seller_id")
            .merge(self.get_active_dates(), on="seller_id")
            .merge(self.get_quantity(), on="seller_id")
            .merge(self.get_sales(), on="seller_id")
        )

        if self.get_review_score() is not None:
            training_set = training_set.merge(self.get_review_score(), on="seller_id")

        return training_set
