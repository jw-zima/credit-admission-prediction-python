def compute_counts_and_share(df, var):
    """Create a summary table with category counts and share
    Args:
    df (data.frame): dataset
    var (string): name of selected variable
    Yields:
    data.frame: table presenting all categories within given feature along with
    count and share
    Examples:
    >>> compute_counts_and_share(df_merged, 'customer_status')
    """
    x1 = df[var].value_counts().to_frame('count')
    x2 = df[var].value_counts(normalize=True).to_frame('share')
    return x1.join(x2)


def compute_variation_coef(x):
    """Compute coefficient of variation (CV)
    Args:
    x (series): numeric series
    Yields:
    numeric: computed CV coefficient
    Examples:
    >>> compute_variation_coef(df_merged.amt_income_total)
    """
    return (x.var())**(0.5) / x.mean()


def convert_object_cols_to_category(df):
    """Convert all 'object' columns to a 'category'
    Args:
    df (data.series): dataset
    Yields:
    data.series: input dataset with converted 'object' columns to 'category'
    Examples:
    >>> convert_object_cols_to_category(df_merged)
    """
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'object' or col_type.name == 'category':
            df[col] = df[col].astype('category')
    return df
