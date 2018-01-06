"""
Python module for scripting helper functions
"""
from glob import glob
import configparser
import os
import re
import json
import pandas as pd


def set_parameter(parameters_filepath, section_name, parameter_name, parameter_value):
    """
    set the specified parameter to the specified value and write back to the *.ini file
    :param parameters_filepath: filename (absolute path)
    :param section_name: section name under which parameter is
    :param parameter_name: parameter name
    :param parameter_value: target value
    :return:
    """
    conf_parameters = configparser.ConfigParser()
    conf_parameters.read(parameters_filepath, encoding="UTF-8")
    conf_parameters.set(section_name, parameter_name, parameter_value)
    with open(parameters_filepath, 'w') as config_file:
        conf_parameters.write(config_file)


def df_2_tex(df, filepath):
    """
    writes a df to tex file
    :param df: dataframe to be converted into tex table
    :param filepath: tex filepath
    :return:
    """
    tex_prefix = r"""\documentclass{standalone}
    \usepackage{booktabs}
    \begin{document}"""

    tex_suffix = r"""\end{document}"""

    with open(filepath, "w") as f:
        f.write(tex_prefix)
        f.write(df.to_latex(float_format="%.1f"))
        f.write(tex_suffix)


def file_rank(filename):
    """
    assign a rank to the file can be used for sorting
    :param filename:
    :return:
    """
    order = {'natural': 0, 'rfgsm_k': 2, 'dfgsm_k': 1, 'bga_k': 3, 'bca_k': 4, 'grosse': 5}

    training_method = re.search("\[training:.*\|", filename).group(0)[:-1].split(':')[-1]
    evasion_method = re.search("\|evasion:.*\]", filename).group(0)[:-1].split(':')[-1]

    return order[training_method] * 6 + order[evasion_method]


def create_tex_tables(filespath="../result_files"):
    """
    Create TeX tables from the results populated under `result_files`
    which is generated from running `framework.py`
    The tex file is stored in `result_files`
    :param filespath: the path where the results in json are stored and the tex files are created
    :return:
    """

    # read the bscn files
    bscn_files = sorted(glob(os.path.join(filespath, "*.txt")), key=lambda x: file_rank(x))

    # read the results file
    files = sorted(glob(os.path.join(filespath, "*.json")), key=lambda x: file_rank(x))

    # dataframes
    bscn_df = pd.DataFrame()
    evasion_df = pd.DataFrame()
    accuracy_df = pd.DataFrame()
    afp_df = pd.DataFrame()
    bon_accuracy_df = pd.DataFrame()
    mal_accuracy_df = pd.DataFrame()
    mal_loss_df = pd.DataFrame()

    for idx, filename in enumerate(bscn_files):
        training_method = re.search("\[training:.*\|", filename).group(0)[:-1].split(':')[-1]
        evasion_method = re.search("\|evasion:.*\]", filename).group(0)[:-1].split(':')[-1]
        with open(filename, 'r') as f:
            bscn_val = float(f.read())
            print(training_method, evasion_method, bscn_val)
            bscn_df.loc[training_method, "bsn_ratio"] = bscn_val

    bscn_df = bscn_df.div(bscn_df.loc['natural'], axis=1)
    for idx, filename in enumerate(files):
        training_method = re.search("\[training:.*\|", filename).group(0)[:-1].split(':')[-1]
        evasion_method = re.search("\|evasion:.*\]", filename).group(0)[:-1].split(':')[-1]
        with open(filename, 'r') as f:
            metrics = json.load(f)
            evasion_df.loc[training_method, evasion_method] = (
                1 - metrics["mal"]["evasion"]["total_correct"] / metrics["mal"]["total"]) * 100.
            if training_method == evasion_method:
                afp_df.loc[training_method, 'accuracy'] = (
                    metrics["mal"]["total_correct"] + metrics["bon"]["total_correct"]) * 100. / (
                        metrics["mal"]["total"] + metrics["bon"]["total"])
                fp = (metrics["bon"]["total"] - metrics["bon"]["total_correct"])
                fn = (metrics["mal"]["total"] - metrics["mal"]["total_correct"])
                tp = (metrics["mal"]["total_correct"])
                tn = (metrics["bon"]["total_correct"])
                afp_df.loc[training_method, 'fpr'] = fp * 100. / (fp + tn)
                afp_df.loc[training_method, 'fnr'] = fn * 100. / (fn + tp)

            accuracy_df.loc[training_method, evasion_method] = (
                metrics["mal"]["total_correct"] + metrics["bon"]["total_correct"]) * 100. / (
                    metrics["mal"]["total"] + metrics["bon"]["total"])

            bon_accuracy_df.loc[training_method, evasion_method] = (
                metrics["bon"]["total_correct"]) * 100. / (metrics["bon"]["total"])
            mal_accuracy_df.loc[training_method, evasion_method] = (
                metrics["mal"]["total_correct"]) * 100. / (metrics["mal"]["total"])
            mal_loss_df.loc[training_method, evasion_method] = metrics["mal"]["total_loss"] / (
                metrics["mal"]["total"])

    # tex file names
    bscn_tbl_file = os.path.join(filespath, "bscn_tbl.tex")
    evasion_tbl_file = os.path.join(filespath, "evasion_tbl.tex")
    accuracy_tbl_file = os.path.join(filespath, "accuracy_tbl.tex")
    afp_tbl_file = os.path.join(filespath, "afp_tbl.tex")
    bon_accuracy_tbl_file = os.path.join(filespath, "bon_accuracy_tbl.tex")
    mal_accuracy_tbl_file = os.path.join(filespath, "mal_accuracy_tbl.tex")
    mal_loss_tbl_file = os.path.join(filespath, "mal_loss_tbl.tex")
    # write the tex files
    df_2_tex(bscn_df, bscn_tbl_file)
    df_2_tex(evasion_df, evasion_tbl_file)
    df_2_tex(accuracy_df, accuracy_tbl_file)
    df_2_tex(afp_df, afp_tbl_file)
    df_2_tex(bon_accuracy_df, bon_accuracy_tbl_file)
    df_2_tex(mal_accuracy_df, mal_accuracy_tbl_file)
    df_2_tex(mal_loss_df, mal_loss_tbl_file)


if __name__ == '__main__':
    create_tex_tables()
