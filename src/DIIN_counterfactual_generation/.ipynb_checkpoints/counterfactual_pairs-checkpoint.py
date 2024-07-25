import pandas as pd

class gen_counterfactual_pairs:
    # def __init__(self, attribute_list1, attribute_list2):
        # self.attribute_list1 = attribute_list1
        # self.attribute_list2 = attribute_list2
        # self.attribute_pair = f"{self.attribute_list1[0]}_{self.attribute_list2[0]}"


    def output_counterfactual_pairs(self, df_00, df_11):
        # df_00 = pd.read_csv('outputs/attribute1_full_pairs_'+self.attribute_pair+'.csv')
        # df_11 = pd.read_csv('outputs/attribute2_full_pairs_'+self.attribute_pair+'.csv')
        
        
        df_00 = df_00.rename(columns={'original_old_word': 'female', 'new_word': 'male'})
        df_11 = df_11.rename(columns={'new_word': 'female', 'original_old_word': 'male'})
        
        df_00 = df_00[['female','male','logits']]
        df_11 = df_11[['female','male','logits']]
        
        
        df_0 = df_00[~df_00.female.isin(df_11.male)]
        df_0
        df_1 = df_11[~df_11.male.isin(df_00.female)]
        df_1
        
        
        
        df_out = pd.merge(df_0,df_1, how='inner', on=['female','male'])
        df_out = df_out.drop_duplicates(subset=['female', 'male','logits_x'], keep='last')
        df_out
        
        
        female_final_df = (df_out.groupby('female').male.apply(pd.Series.mode)).reset_index()
        female_final_df = female_final_df[female_final_df["level_1"]==0]
        female_final_df
        
        
        male_final_df = (df_out.groupby('male').female.apply(pd.Series.mode)).reset_index()
        male_final_df = male_final_df[male_final_df["level_1"]==0]
        male_final_df
        
        
        
        count = 0
        male_final = []
        for index, row in male_final_df.iterrows():
            # try:
            if (row['male'] != row['female']):
                male_final.append([row['male'], row['female']])
                count+=1
                print(row['male'], row['female'],count)
            # except:
            #     (row['male'], row['female'])
        male_final = pd.DataFrame(male_final)
        # male_final = male_final.rename(columns={0: 'female', 1: 'male'})
        male_final
        
        
        
        count = 0
        female_final = []
        for index, row in female_final_df.iterrows():
            # try:
            if (row['female'] != row['male']):
                female_final.append([row['female'], row['male']])
                count+=1
                print(row['female'], row['male'],count)
            # except:
            #     (row['female'], row['male'])
        female_final = pd.DataFrame(female_final)
        # female_final = female_final.rename(columns={0: 'female', 1: 'male'})
        female_final
        
        
        
        final_df = pd.concat([female_final, male_final])
        final_df = final_df.rename(columns={0: 'original', 1: 'counterfactual'})
        final_df
        # final_df.to_csv('outputs/df1.csv')
        
        # filter = final_df['original']=="i"
        # final_df = final_df[~filter]
        filter = final_df['original'].str.contains("-")
        final_df = final_df[~filter]
        # filter = df['original_old_word'].str.contains("[SEP]")
        # df = df[~filter]
        # filter = df['original_old_word'].str.contains("[PAD]")
        # df = df[~filter]
        final_df
        
        
        final_df = final_df[~final_df.original.str.contains(r'\d')]
        final_df = final_df[~final_df.counterfactual.str.contains(r'\d')]
        filter = final_df['original'].str.contains("world")
        final_df = final_df[~filter]
        final_df
        
        
        # final_df2 = final_df.copy()
        # final_df2['original'] = final_df['original'] + '.'
        # final_df2['counterfactual'] = final_df['counterfactual'] + '.'
        # final_df2
        
        
        # final_df = pd.concat([final_df,final_df2])
        final_df
        
        print(final_df.to_string())
        
        final_df.to_csv('outputs/counterfactuals_df.csv', index=False)