import pandas as pd
import numpy as np
import os

def MATCH(lookup_values, lookup_array , match_type=1):
    '''Excel MATCH in Python form'''
    pos = []
    if match_type == 1:
        ascend_array = np.sort(lookup_array)
        for i in lookup_values:
            for j, x in enumerate(ascend_array):
                if x > i:
                    pos.append(lookup_array.tolist().index(ascend_array[j]))
                    break
                    
    elif match_type == 0:
        for i in lookup_values:
            for j in range(len(lookup_array)):
                if i == lookup_array[j]:
                    pos.append(j)
                    break
                    
    elif match_type == -1:
        descend_array = np.sort(lookup_array)[::-1]
        for i in lookup_values:
            for j, x in enumerate(descend_array):
                if x < i:
                    pos.append(lookup_array.tolist().index(descend_array[j]))       
    else:
        print('match_type takes values 1, 0, or -1')
            
    if pos == []: return '#N/A'
    else: return pos
    
    
def MODE(lookup_array):
    '''Excel MODE in Python form'''
    occur = []
    for i in lookup_array:
        o = 0
        for j in lookup_array:
            if i == j: o += 1
        occur.append(o)    
    return lookup_array[occur.index(max(occur))]


def INDEX(lookup_array, row_number):
    '''Excel INDEX in Python form'''
    return lookup_array[row_number]
    

def COUNTIF(lookup_array, value):
    '''Excel COUNTIF in Python form'''
    count = 0
    for i in lookup_array:
        if i == value:
            count += 1
    return count


def REINDEX(array):
    '''Sorts sentences by constraint in asencing order'''
    array_ascend = np.sort(array)
    unique, idx = [], []
    for i, x in enumerate(array_ascend):
        if x not in unique:
            idx.append(np.where(array==x)[0].tolist())
            unique.append(x)
    return sum(idx, [])


def concat_best(df, filename, N):
    '''Concatenates all the best completions provided by offline participants.
       df: an instance of Pandas.DataFrame, all the sentences frames in the list;
       filename: each participants answer sheet should start with the same file name;
       N: number of participants.'''
    best = ['Best'+str(i+1) for i in range(N)]
    for i in range(1, N+1):
        try:
            subj_ans = pd.read_excel(filename + ' (' + str(i) + ').xlsx', sheet_name='問卷內容')
        except ValueError:
            try: 
                subj_ans = pd.read_excel(filename + ' (' + str(i) + ').xlsx', sheet_name='工作表1')
            except ValueError:
                print('Please provide the correct sheet name.')
        except FileNotFoundError:
            print('Please provide the correct path to the file.')
            
        df.insert(len(df.columns), best[i-1], subj_ans['最好的結尾 (必填)'])
    return df

def concat_ans_percent(df, N, start):
    '''df: an instance of Pandas.DataFrame with all the best completions concatenated (e.g., an object concet_best returns);
       N: number of participants;
       start: the column index of the first best completion'''
    percent, optimal, constraint, col = {}, [], [], ['Ans%_'+str(i+1) for i in range(N)]
    
    for i in range(len(df)):
        ans_all = [x for x in df.iloc[i, start:start+N]]
        ans = list(filter(pd.notna, ans_all))
        percent[i] = [COUNTIF(ans, ans_all[j])/len(ans) for j in range(len(ans_all))]
        opt = INDEX(ans, MODE(MATCH(ans, ans, match_type=0)))
        optimal.append(opt)
        constraint.append(COUNTIF(ans, opt)/len(ans))
        
    df_percent = pd.DataFrame(percent, index=col).transpose()
    df = df.assign(Optimal=optimal, Constraint=constraint)
    df_ans = pd.concat([df, df_percent], axis=1)
    
    return df_ans


def concat_alts(df_ans, filename, N):
    '''Concatenates all the alternative answers provided by offline participants.
       df: an instance of Pandas.DataFrame (e.g., an object concat_ans_percent returns);
       filename: each participants answer sheet should start with the same file name;
       N: number of participants.'''
    second, third = ['Second'+str(i+1) for i in range(N)], ['Third'+str(i+1) for i in range(N)]
    
    for i in range(1, N+1):
        try:
            subj_ans = pd.read_excel(filename + ' (' + str(i) + ').xlsx', sheet_name='問卷內容')
        except ValueError:
            try: 
                subj_ans = pd.read_excel(filename + ' (' + str(i) + ').xlsx', sheet_name='工作表1')
            except ValueError:
                print('Please provide the correct sheet name.')
        except FileNotFoundError:
            print('Please provide the correct path to the file.')
    
        df_ans.insert(len(df_ans.columns), second[i-1], subj_ans['結尾2 (必填)'])
    
    for i in range(1, N+1):
        try:
            subj_ans = pd.read_excel(filename + ' (' + str(i) + ').xlsx', sheet_name='問卷內容')
        except ValueError:
            try: 
                subj_ans = pd.read_excel(filename + ' (' + str(i) + ').xlsx', sheet_name='工作表1')
            except ValueError:
                print('Please provide the correct sheet name.')
        except FileNotFoundError:
            print('Please provide the correct path to the file.')
    
        df_ans.insert(len(df_ans.columns), third[i-1], subj_ans['結尾3'])
    
    return df_ans


if __name__ == "__main__":
    os.chdir('/home/amandalin047/cloze_2023/cloze_ratings_march')
    N = 15
    template1, template2 = pd.read_excel('template1.xlsx'), pd.read_excel('template2.xlsx')
    filename1, filename2 = 'Cloze Rating0305_1', 'Cloze Rating0305_2'

    List1, List2 = concat_best(template1, filename1, N), concat_best(template2, filename2, N)
    List1_ans, List2_ans = concat_ans_percent(List1, N, 2), concat_ans_percent(List2, N, 2)
    List1_final, List2_final = concat_alts(List1_ans, filename1, N), concat_alts(List2_ans, filename2, N)

    Cloze = pd.concat([List1_final, List2_final], axis=0)
    Cloze = Cloze.reset_index(drop=True)
    reindex = REINDEX(Cloze['Constraint'].to_numpy())
    Cloze = Cloze.reindex(reindex)
    Cloze.to_excel('March_python.xlsx', index=False)
