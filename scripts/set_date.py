# empty list to hold date strings
def get_time_out_idx(df):
    indexes = []
    for df_dates in df.index:
        string_df = f"{df_dates}" # Convert the Timestamp to string
        split_string_df = string_df.split(' ')
        df_dates = split_string_df[0]
        indexes.append(df_dates)
    df.index = indexes
    return df