import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.pyplot import figure
from lifetimes.plotting import plot_history_alive





def plot_proba_alive(customer_id, df_orders, df_rfmt, model):
    #figure(num=None, figsize=(28, 3), dpi=80, facecolor='w', edgecolor='k')
    fig, ax = plt.subplots(figsize=(20, 10), nrows=1, ncols=1)  # create figure & 1 axis
    example_customer_orders = df_orders.loc[df_orders['customer_id'] == customer_id]
    days_since_birth = df_rfmt.loc[customer_id, "T"].astype(int)
    ax = plot_history_alive(model, days_since_birth, example_customer_orders, 'date_created')
    ax.properties()['children'][0].set_color('#28ba73')
    ax.properties()['children'][0].set_linewidth(3)
    ax.properties()['children'][1].set_color('#6E6F73')
    ax.set_facecolor("#f2f2f2ff")
    ax.set_title('History of probability of still being a customer')
    ax.set_ylabel('P(still customer)')
    ax.legend()
    return fig


def plot_purchases_history(df_orders, customer_id):
    df_orders_cust = df_orders[df_orders.customer_id == customer_id]
    df_orders_cust = df_orders_cust.sort_values(by="date_created")
    per = df_orders_cust.date_created.dt.to_period("M")
    g = df_orders_cust.groupby(per).agg({"total_revenue":"sum", "quantity":"sum"}).reset_index()
    g.date_created = g.date_created.astype('datetime64[ns]') 
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=g.date_created, y=g.total_revenue, name="Amount spent",line_shape='spline', marker_color='#28ba73'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=g.date_created, y=g.quantity, name="Quantity purchased", line_shape='spline', marker_color='#d4df33'),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Amount spent", secondary_y=False)
    fig.update_yaxes(title_text="Quantity purchased", secondary_y=True, showgrid=False)
    fig.update_layout(width=1050, height=450, plot_bgcolor='#f2f2f2')
    return fig


#taken from https://github.com/CamDavidsonPilon/lifetimes/blob/0a0a84fe4b10fff0bdaa6a6020d930c8dc6aee2d/lifetimes/utils.py
def calculate_alive_path(
    model,
    transactions,
    datetime_col,
    t,
    freq="D"
):
    """
    Calculate alive path for plotting alive history of user.
    Uses the ``conditional_probability_alive()`` method of the model to achieve the path.
    Parameters
    ----------
    model:
        A fitted lifetimes model
    transactions: DataFrame
        a Pandas DataFrame containing the transactions history of the customer_id
    datetime_col: string
        the column in the transactions that denotes the datetime the purchase was made
    t: array_like
        the number of time units since the birth for which we want to draw the p_alive
    freq: string, optional
        Default: 'D' for days. Possible values listed here:
        https://numpy.org/devdocs/reference/arrays.datetime.html#datetime-units
    Returns
    -------
    :obj: Series
        A pandas Series containing the p_alive as a function of T (age of the customer)
    """

    customer_history = transactions[[datetime_col]].copy()
    customer_history[datetime_col] = pd.to_datetime(customer_history[datetime_col])
    customer_history = customer_history.set_index(datetime_col)
    # Add transactions column
    customer_history["transactions"] = 1

    # for some reason fillna(0) not working for resample in pandas with python 3.x,
    # changed to replace
    purchase_history = customer_history.resample(freq).sum().replace(np.nan, 0)["transactions"].values

    extra_columns = t + 1 - len(purchase_history)
    customer_history = pd.DataFrame(np.append(purchase_history, [0] * extra_columns), columns=["transactions"])
    # add T column
    customer_history["T"] = np.arange(customer_history.shape[0])
    # add cumulative transactions column
    customer_history["transactions"] = customer_history["transactions"].apply(lambda t: int(t > 0))
    customer_history["frequency"] = customer_history["transactions"].cumsum() - 1  # first purchase is ignored
    # Add t_x column
    customer_history["recency"] = customer_history.apply(
        lambda row: row["T"] if row["transactions"] != 0 else np.nan, axis=1
    )
    customer_history["recency"] = customer_history["recency"].fillna(method="ffill").fillna(0)

    return customer_history.apply(
        lambda row: model.conditional_probability_alive(row["frequency"], row["recency"], row["T"]), axis=1
    )


