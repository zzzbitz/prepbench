
import pandas as pd
from pathlib import Path


def solve(inputs_dir: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(inputs_dir / "input_01.csv")

    df['Price'] = df['Price'].fillna(1.5)
    df['MemberID'] = df['MemberID'].fillna(0).astype(int)

    df_pivot = df.pivot_table(
        index=['TicketID', 'MemberID'],
        columns='Type',
        values='Price',
        aggfunc=['count', 'sum']
    ).fillna(0)

    df_pivot.columns = ['_'.join(col).strip()
                        for col in df_pivot.columns.values]
    df_pivot.reset_index(inplace=True)

    df_pivot.rename(columns={
        'count_Drink': 'Drink_Count',
        'count_Main': 'Main_Count',
        'count_Snack': 'Snack_Count',
        'sum_Drink': 'Drink_Sum',
        'sum_Main': 'Main_Sum',
        'sum_Snack': 'Snack_Sum'
    }, inplace=True)

    df_pivot['Number of Meal Deals'] = df_pivot[['Drink_Count',
                                                 'Main_Count', 'Snack_Count']].min(axis=1).astype(int)
    df_pivot['Total Ticket Price'] = df_pivot['Drink_Sum'] + \
        df_pivot['Main_Sum'] + df_pivot['Snack_Sum']

    df_filtered = df_pivot[df_pivot['Number of Meal Deals'] > 0].copy()

    df_filtered['Avg_Drink_Price'] = (
        df_filtered['Drink_Sum'] / df_filtered['Drink_Count']).fillna(0)
    df_filtered['Avg_Main_Price'] = (
        df_filtered['Main_Sum'] / df_filtered['Main_Count']).fillna(0)
    df_filtered['Avg_Snack_Price'] = (
        df_filtered['Snack_Sum'] / df_filtered['Snack_Count']).fillna(0)

    df_filtered['Total Meal Deal Earnings'] = df_filtered['Number of Meal Deals'] * 5

    excess_drink_value = (
        df_filtered['Drink_Count'] - df_filtered['Number of Meal Deals']) * df_filtered['Avg_Drink_Price']
    excess_main_value = (
        df_filtered['Main_Count'] - df_filtered['Number of Meal Deals']) * df_filtered['Avg_Main_Price']
    excess_snack_value = (
        df_filtered['Snack_Count'] - df_filtered['Number of Meal Deals']) * df_filtered['Avg_Snack_Price']

    df_filtered['Total Excess'] = (
        excess_drink_value + excess_main_value + excess_snack_value).round(2)

    df_filtered['Ticket Price Variance to Meal Deal Earnings'] = df_filtered['Total Ticket Price'] - \
        (df_filtered['Total Meal Deal Earnings'] + df_filtered['Total Excess'])

    output_df = df_filtered[[
        'Total Ticket Price',
        'Ticket Price Variance to Meal Deal Earnings',
        'Total Meal Deal Earnings',
        'Total Excess',
        'TicketID',
        'MemberID'
    ]]

    return {"output_01.csv": output_df}


if __name__ == "__main__":
    task_dir = Path(__file__).parent
    inputs_dir = task_dir / "inputs"
    cand_dir = task_dir / "cand"

    if not cand_dir.exists():
        cand_dir.mkdir()

    outputs = solve(inputs_dir)

    for filename, df in outputs.items():
        df.to_csv(cand_dir / filename, index=False, encoding='utf-8')
