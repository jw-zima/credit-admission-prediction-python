from plotnine import *
import pandas as pd


def plot_histogram_per_status(df, cols, use_boarder_valus=True):
    """Make a hisrogram using ggplot grammar of graphics with possibility to
    remove boarder values (0 and 1)
    Args:
    df (data.frame): dataset
    cols (list): list with names of status variables for density estimation
    use_boarder_valus (boolean): disposition to remove boarder values
    Yields:
    plot: Density plots with distribution or each status variable
    Examples:
    >>> plot_histogram_per_status(df, df.columns, True)
    """
    for col in cols:
        if use_boarder_valus is not True:
            df = df.loc[(df[col] > 0) & (df[col] < 1), :]
        print(ggplot(df)
              + geom_histogram(aes(x=col))
              + theme_light()
              + xlab("status: " + col)
              )


def plot_boxplot_by_target(df, num_var, target_var, outliers_to_remove=None):
    """Make a box plot split by target using ggplot grammar of graphics
    Args:
    df (data.frame): dataset
    num_var (string): name of numeric variable for density estimation
    cluster_var (string): name of variable with target
    Yields:
    plot: Density plot for distributions split by target
    Examples:
    >>> plot_boxplot_by_target(dataset, "age", "target")
    """
    df[target_var] = df[target_var].astype('object')

    if not(pd.isnull(outliers_to_remove)):
        if (type(outliers_to_remove) == float):
            quantiles = df[num_var].quantile([outliers_to_remove,
                                              1 - outliers_to_remove])
            df = df.loc[((df[num_var] > quantiles.iloc[0])
                        & (df[num_var] < quantiles.iloc[1])), :]
    try:
        g = (ggplot(df)
             + geom_boxplot(aes(x=target_var, y=num_var, fill=target_var))
             + ggtitle(num_var)
             + theme_light()
             + scale_fill_brewer(type="qual", palette="Dark2"))
        return print(g)
    except Exception:
        pass


def plot_share_of_categorical_vars_by_target(df, var, target_var):
    """Make a column plot colored by clusters using ggplot grammar of graphics
    to present share of given binery/categorical var split by target
    Args:
    df (data.frame): dataset
    var (string): name of variable for x axis
    cluster_var (string): name of variable with target
    Yields:
    plot: Column plot with share of each category in given categroical variable
    split by target
    Examples:
    >>> plot_share_of_categorical_vars_by_target(dataset, "flag_own_car",
                                               "target")
    """
    df = df.loc[:, [var, target_var]]
    df = df.groupby([var, target_var]).value_counts()
    df = df.reset_index().rename(columns={0: 'n'})
    df['n_cluster'] = df.groupby([target_var]).n.transform('sum')
    df['share'] = round(100 * df['n'] / df['n_cluster'], 1)

    df[target_var] = df[target_var].astype('object')
    df[var] = df[var].astype('object')

    g = (ggplot(df)
         + geom_col(aes(x=var, y="share", fill=target_var))
         + facet_wrap('~' + target_var)
         + theme_light()
         + theme(legend_position="top")
         + theme(axis_text_x=element_text(rotation=90, hjust=1))
         + scale_fill_brewer(type="qual", palette="Dark2"))
    print(g)
