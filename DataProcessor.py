import pandas as pd
import re
import json
import random


def papers_by_single_category(category: str, papers: list) -> list:
    categories = set([el['categories'] for el in papers])
    cat_wanted_only = []
    for el in categories:
        cats = el.split(' ')
        in_el_flg = True
        for subel in cats:
            if not re.search(category, subel, re.IGNORECASE):
                in_el_flg = False
        if in_el_flg:
            cat_wanted_only.append(el)
    # cat_wanted_only = [el for el in categories if re.search(category, el, re.IGNORECASE) and not re.search(' ', el)]
    for el in cat_wanted_only:
        # print(el)
        pass
    papers_wantedonly = []
    for el in papers:
        if el['categories'] in cat_wanted_only:
            papers_wantedonly.append(el)
    return papers_wantedonly


def class_balancer(dataframes: list, size: int) -> pd.DataFrame:
    """
    Balances classes by random selection with replacement
    :param dataframes: list of pd.Dataframes each one representing 1 class
    :param size: size of output pd.Dataframe
    :return: shuffled pd.Dataframe with equal class representation
    """
    classes = len(dataframes)
    output_data = pd.DataFrame()
    for i in range(size):
        cat = random.randint(0, classes - 1)
        el = random.randint(0, len(dataframes[cat].index) - 1)
        test = dataframes[cat].iloc[el, :]
        output_data = pd.concat([output_data, pd.DataFrame(dataframes[cat].iloc[el, :]).T])
    output_data.reset_index(drop=True, inplace=True)
    return output_data


def process():
    pd.set_option('display.max_columns', None)
    file = r'arxiv-metadata-oai-snapshot.json'
    data = []
    count = 0
    wanted_keys = ['categories', 'title', 'abstract', 'id']
    write_file = 'daten_bsp.json'
    write_lines = []
    print('read 10 lines')
    for line in open(file):
        if count > 10:
            break
        write_lines.append(line)
        count += 1

    print('write 10 lines')
    with open(write_file, 'w') as outfile:
        for line in write_lines:
            outfile.write(line)

    print('read 10000 lines')
    for line in open(file):
        if count > 10000:
            break
        line_dict = json.loads(line)
        for key in line_dict:
            if key not in wanted_keys:
                tmp = dict(line_dict)
                del tmp[key]
                line_dict = tmp
        data.append(line_dict)
        count += 1
    print('sort papers')
    papers_physonly = papers_by_single_category('ph', data)
    # print(len(papers_physonly))
    papers_mathonly = papers_by_single_category('math', data)
    # print(len(papers_mathonly))
    papers_csonly = papers_by_single_category('^cs', data)
    # print(len(papers_csonly))
    papers_econonly = papers_by_single_category('econ', data)
    # print(len(papers_econonly))
    # 0 econ only papers
    papers_nlinonly = papers_by_single_category('nlin', data)
    # print(len(papers_nlinonly))
    papers_qbioonly = papers_by_single_category('q-bio', data)
    # print(len(papers_qbioonly))

    categories = set([el['categories'] for el in data])
    categories = sorted(categories)
    # for el in categories:
    #     #print(el)
    #     pass

    data_physonly = pd.DataFrame([x['abstract'] for x in papers_physonly])
    data_physonly['class'] = (['physics' for x in range(len(data_physonly))])
    data_mathonly = pd.DataFrame([x['abstract'] for x in papers_mathonly])
    data_mathonly['class'] = (['math' for x in range(len(papers_mathonly))])
    data_nlinonly = pd.DataFrame([x['abstract'] for x in papers_nlinonly])
    data_nlinonly['class'] = (['nlin' for x in range(len(papers_nlinonly))])
    data_csonly = pd.DataFrame([x['abstract'] for x in papers_csonly])
    data_csonly['class'] = (['cs' for x in range(len(papers_csonly))])
    data_qbioonly = pd.DataFrame([x['abstract'] for x in papers_qbioonly])
    data_qbioonly['class'] = (['qbio' for x in range(len(papers_qbioonly))])

    data = data_physonly.append(data_mathonly)
    data = data.append(data_csonly)
    data = data.append(data_csonly)
    data = data.append(data_csonly)
    data.reset_index(drop=True, inplace=True)

    print('balancing data')
    # Data balanced for equal class representation
    equal_data = class_balancer([data_physonly, data_mathonly, data_csonly, data_nlinonly, data_qbioonly], 10000)
    equal_data.columns = ['abstract', 'class']
    data = equal_data

    return data
