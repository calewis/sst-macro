import pandas as pd
import numpy as np
import os

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from compiledtrees import code_gen


def csvs_to_dataframes(csv_files):
    return pd.concat([pd.read_csv(x) for x in csv_files])


def get_arg_cols(df):
    return [x for x in df.keys() if x.startswith('arg')]


def extract_medians(df, arg_cols=None, target_field="time"):
    if arg_cols is None:
        arg_cols = get_arg_cols(df)

    return df.groupby(arg_cols)[target_field].median().reset_index()


def add_percentiles(df, arg_cols=None, target_field="time"):
    def generate_percentiles(df):
        return list(zip(
                list(range(0, 10, 1)),
                np.percentile(df, list(range(0, 100, 10)))))

    if arg_cols is None:
        arg_cols = get_arg_cols(df)

    df = df.groupby(arg_cols)[target_field] \
           .apply(generate_percentiles) \
           .reset_index() \
           .explode(target_field)

    df[["argp", target_field]] = pd.DataFrame(
            df[target_field].tolist(),
            index=df.index)

    return df


def split_test_train(df, train_size=0.8, arg_cols=None, target_field='time'):
    if arg_cols is None:
        arg_cols = get_arg_cols(df)

    return train_test_split(df[arg_cols], df[target_field],
                            train_size=train_size)


def train_ETR_model(X, y, n_jobs=-1, criterion="mse", n_estimators=10):
    model = ExtraTreesRegressor(n_jobs=n_jobs,
                                criterion=criterion, n_estimators=n_estimators)
    model.fit(X, y)
    return model


def train_RF_model(X, y, n_jobs=-1, criterion="mse", n_estimators=10):
    model = RandomForestRegressor(n_jobs=n_jobs,
                                  criterion=criterion,
                                  n_estimators=n_estimators)
    model.fit(X, y)
    return model


def clean_up_code(lines, new_name=None):
    def clean_line(line):
        line = line.lstrip()
        if "evaluate" in line:
            if "__attribute__((__always_inline__))" in line:
                line = line.replace("__attribute__((__always_inline__))", "")
            elif "+=" not in line:  # This is the declaration
                line = "__attribute__((pure)) " + line
        if new_name:
            line = line.replace("evaluate", new_name)
        return line

    return [clean_line(x) for x in lines]


def write_cmakelist(output_dir, libname=None):
    cmake_text = """cmake_minimum_required(VERSION 3.12)
    project(SSTrees CXX)
    file(GLOB SOURCES "*.cpp")
    add_library(ssttrees MODULE ${SOURCES})
    """

    if libname:
        cmake_text = cmake_text.replace("ssttrees", libname)

    with open(os.path.join(output_dir, "CMakeLists.txt"), 'w') as f:
        f.write(cmake_text)


def compile_forest_to_files(model, name, output_dir):
    files = code_gen.code_gen_ensemble(
            trees=[e.tree_ for e in model.estimators_],
            individual_learner_weight=1.0/model.n_estimators,
            initial_value=0.0, n_jobs=-1)

    files_as_lines = []
    for f in files:
        f.seek(0)
        files_as_lines.append(
                clean_up_code(
                    [x.decode("utf-8").rstrip() for x in f.readlines()],
                    name
                ))

    i = 0
    for filelines in files_as_lines:
        if any("+=" in line for line in filelines):
            with open(os.path.join(output_dir, name + ".cpp"), 'w') as f:
                f.writelines(filelines)
        else:
            with open(os.path.join(output_dir, name + "_{}.cpp".format(i)), 'w') as f:
                f.writelines(filelines)
            i = i + 1

    return files_as_lines
