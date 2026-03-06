import streamlit as st

# st.title("new")

Stock_Bought_Price = st.sidebar.number_input(
    "Stock Bought Price",
    min_value=1.0,
)
Stock_Bought_Quantity = st.sidebar.number_input("Stock Bought Quantity", min_value=1)
New_Market_Price = st.sidebar.number_input("New Market Price", min_value=1.0)
New_Share_Bought = st.sidebar.number_input("Share to be Bought", min_value=1)


# stock average checking
def stock_average(
    previous_bought_price: float,
    previous_share_count: int,
    new_share_price: float,
    new_share_count: int,
):
    total_shares = previous_bought_price * previous_share_count
    total_money = new_share_price * new_share_count
    # print(f"New Money to Invest in - ₹{total_money}")

    avg = (total_shares + total_money) / (previous_share_count + new_share_count)
    # print(f"New Avg stock value - ₹{round(avg, 2)}")
    return total_money, avg


def stock_sell_checker(
    previous_bought_price: float, previous_share_count: int, new_sell_price: float
):
    total_invested = previous_bought_price * previous_share_count
    # print(f"Money in investment - ₹{total_invested}")
    profit = round((new_sell_price * previous_share_count) - total_invested, 2)
    profit_percent = round(
        ((new_sell_price - previous_bought_price) / previous_bought_price) * 100, 2
    )
    # print(f"If you sell at ₹{new_sell_price}\nThen you would earn ₹{profit}", end=" - ")
    # print(f"A total of {profit_percent}% profit")

    return total_invested, profit, profit_percent


t_money, average = stock_average(
    Stock_Bought_Price, Stock_Bought_Quantity, New_Market_Price, New_Share_Bought
)

t_invested, p, pp = stock_sell_checker(
    Stock_Bought_Price, Stock_Bought_Quantity, New_Market_Price
)


generate = st.sidebar.button("generate")

if generate:
    st.write(f"New Money to Invest in - ₹{t_money}")
    st.write(f"New Avg stock value - ₹{round(average, 2)}")

    st.write(f"Money in investment - ₹{t_invested}")
    st.write(f"If you sell at ₹{New_Market_Price}\nThen you would earn ₹{p} - A total of {pp}% profit")

