d={'HR': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
 'O2Sat': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
 'Temp': [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
 'MAP': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
 'Resp': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
 'BUN': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
 'Chloride': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
 'Creatinine': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
 'Glucose': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
 'Hct': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
 'Hgb': [0, 1, 2, 3, 4, 6],
 'WBC': [0, 1, 2, 3, 4, 5, 6, 7, 9],
 'Platelets': [0, 1, 2, 3, 4, 5, 6, 7, 9],
 'Age': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 'Gender': [0, 1]}

tokens={
    'padding_index':0,
}

index=1
for i in d.keys():
  for j in range(len(d[i])):
    if j<len(d[i])-1:
      tokens[f"{i}_{j}"]=index
    else:
      if i!="Gender":
        tokens[f"{i}_nan"]=index
      else:
        tokens[f"{i}_{j}"]=index

    index+=1
        
def impute_row(val,row):
  if val==0.0:
    return get_val(tokens,f"{row}_{0}")
  elif val==1.0:
    return get_val(tokens,f"{row}_{1}")
  elif val==2.0:
    return get_val(tokens,f"{row}_{2}")
  elif val==3.0:
    return get_val(tokens,f"{row}_{3}")
  elif val==4.0:
    return get_val(tokens,f"{row}_{4}")
  elif val==5.0:
    return get_val(tokens,f"{row}_{5}")
  elif val==6.0:
    return get_val(tokens,f"{row}_{6}")
  elif val==7.0:
    return get_val(tokens,f"{row}_{7}")
  elif val==8.0:
    return get_val(tokens,f"{row}_{8}")
  elif val==9.0:
    return get_val(tokens,f"{row}_{9}")
  else:
    return tokens[f'{row}_nan']


def get_val(tokens,val):
  try:
    return tokens[val]
  except:
    row=val.split('_')[0]
    return tokens[f"{row}_nan"]

def impute_logic(df):
  for i in df.columns:
    df[i]=df[i].apply(impute_row,row=i)



