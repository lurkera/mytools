import math
import os
import numba as nb
import pandas as pd

pd.set_option('max_columns', 100, 'max_rows', 100, 'expand_frame_repr', False)

'''
锚点优选功能核查：检查锚点、非锚点LTE小区中配置的频点优先级是否符合要求：
1、FDD1800：7
2、F:6
3、其他：0
'''


@nb.vectorize('int64(int64,int64)', nopython=True, fastmath=True, target='parallel', cache=True)
def check_freq_priority(freq, priority):
    result = 1
    if 1200 <= freq <= 1400:
        if priority != 7: result = 0
    elif 38350 <= freq <= 38550:
        if priority != 6: result = 0
    else:
        if priority != 0: result = 0
    return result


'''
根据经纬度计算距离，主要参数的顺序(纬度1，经度1，纬度2，经度2)
'''


@nb.vectorize('float64(float64,float64,float64,float64)', nopython=True, fastmath=True, target='parallel', cache=True)
def getdistance(lat1, lon1, lat2, lon2):
    lng1 = math.radians(lon1)
    lat1 = math.radians(lat1)
    lng2 = math.radians(lon2)
    lat2 = math.radians(lat2)
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    distance = 2 * math.asin(math.sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance, 3)
    return distance


@nb.vectorize('int64(float64,float64)', nopython=True, fastmath=True, target='parallel', cache=True)
def check_same(p1, p2):
    return 1 if p1 == p2 else 0


@nb.vectorize('int64(int64,int64,int64,int64,int64)', nopython=True, fastmath=True, target='parallel',
              cache=True)
def nsa_handover_check(A3A1, A3A2, A4A1, A4A2, A4):
    result = 1
    if A3A1 != -105 or A3A2 != -108 or A4A1 != -105 or A4A2 != -108 or A4 != -105:
        result = 0
    return result


'''
核查4g锚点配置的ssb频点与5G小区的ssb频点是否一致
1、如果配置的是绝对频点，直接比较
2、如果配置的是ssb相对位置，则根据ssb位置进行对比
    。ssb位置=6321，对应频点为504990
    。ssb位置=6363，对应频点为509070
    。ssb位置=6411，对应频点为512910
    如果对比结果一致则返回1，不一致返回0，如果没有找到ssb相对位置对应的频点则返回2
'''


@nb.vectorize('int64(int64,int64,int64)', nopython=True, fastmath=True, target='parallel', cache=True)
def check_ssbfreq(ssbtype, ssbposition, freq):
    result = None
    if ssbtype == 0:
        result = 0 if ssbposition != freq else 1
    else:
        if ssbposition == 6312:
            result = 0 if freq != 504990 else 1
        elif ssbposition == 6363:
            _freq = 509070
            result = 0 if freq != 509070 else 1
        elif ssbposition == 6411:
            result = 0 if freq != 512910 else 1
        else:
            result = 2
    return result


# 4g
para_4g = {
    '4g工参': [None, ['地市', '小区英文名', 'cgi', '工作频段', '经度', '纬度', '厂家', '覆盖场景']],
    '查询NR邻区关系': [None, ['NAME', '本地小区标识', '基站标识', '小区标识']],
    '查询NR外部小区': [None, ['NAME', '基站标识', '小区标识', '下行频点', '物理小区标识', '跟踪区域码']],
    '查询NSA_DC管理参数配置': [None, ['NAME', '本地小区标识', 'NSA DC算法开关&NSA DC能力开关', '上层指示开关', 'NSA DC算法开关&周期性触发PCC锚点选择开关']],
    '查询NR_SCG频点配置': [None, ['NAME', 'NSA DC B1事件RSRP门限(毫瓦分贝)']],
    '查询主载波频点配置': [None, ['NAME', '主载波下行频点', 'NSA PCC锚点优先级', 'NSA DC主载波A4事件的RSRP触发门限(毫瓦分贝)']],
    '查询异频切换参数组': [None, ['NAME', '本地小区标识', '异频切换参数组ID', '基于A4A5异频A2 RSRP触发门限(毫瓦分贝)', '基于A3的异频A2 RSRP触发门限(毫瓦分贝)',
                         '基于覆盖的异频RSRP触发门限(毫瓦分贝)', '基于A4A5异频A1 RSRP触发门限(毫瓦分贝)', '基于A3的异频A1 RSRP触发门限(毫瓦分贝)']],
    '查询小区QCI参数': [None, ['NAME', '本地小区标识', '服务质量等级', 'NSA DC异频切换参数组ID']],
    '查询小区静态参数': [None, ['NAME', '本地小区标识', '小区名称']],
    '查询小区动态参数': [None, ['NAME', '本地小区标识', '小区的实例状态']],
    '4g异常数据': [None, ['NAME', '执行结果']]
}

# 5g
para_5g = {
    '5g工参': [None, ['所属地市', '小区中文名', '网管中网元名称', '经度', '纬度', '覆盖场景', '厂家']],
    '查询NR小区NSA_DC参数配置': [None, ['NAME', 'NR小区标识', 'PSCell A2事件RSRP门限(毫瓦分贝)']],
    '查询NR_DU小区静态参数': [None, ['NAME', 'NR DU小区标识', 'NR DU小区名称', '小区标识', '物理小区标识', 'SSB频域位置描述方式', 'SSB频域位置']],
    '查询NR小区动态参数': [None, ['NAME', 'NR小区标识', '小区可用状态']],
    '查询gNodeB跟踪区域信息': [None, ['NAME', '跟踪区域码']],
    '查询gNodeB功能': [None, ['NAME', 'gNodeB标识']],
    '查询gNodeB运营商信息': [None, ['NAME', 'NR架构选项']],
    '5g异常数据': [None, ['NAME', '执行结果']]
}

pth = None
pth = r'e:\python\0424'
while not pth:
    pth = input('请输入锚点参数文件目录路径:')
    if pth:
        if not os.path.exists(pth):
            print('输入目录路径不存在，请重新输入')
            pth = None
print('**********程序开始*********')
files = os.listdir(pth)
# 在para_4g字典中key对应得文件能够找到，则将文件路径赋值给key对于的value
for k in para_4g.keys():
    result = filter(lambda x: k in x, files).__next__()
    if result:
        para_4g[k][0] = os.path.join(pth, result)
# 查找未找到的文件，并报错
para_4g_none = [k for k, v in para_4g.items() if v[0] == None]
if para_4g_none:
    print('ERROR:4G参数表 " {} "不存在，请核查数据'.format(','.join(para_4g_none)))
    exit()

# 在para_5g字典中key对应得文件能够找到，则将文件路径赋值给key对于的value
for k in para_5g.keys():
    result = filter(lambda x: k in x, files).__next__()
    if result:
        para_5g[k][0] = os.path.join(pth, result)
para_5g_none = [k for k, v in para_5g.items() if v[0] == None]
# 查找未找到的文件，并报错
if para_5g_none:
    print('ERROR:5G参数表 " {} "不存在，请核查数据'.format(','.join(para_5g_none)))
    exit()
for k, v in para_4g.items():
    print('正在读取4G参数表"{}"......'.format(k))
    para_4g[k] = pd.read_csv(para_4g[k][0], encoding='gbk', low_memory=False, usecols=para_4g[k][1])

for k, v in para_5g.items():
    print('正在读取5G参数表"{}"......'.format(k))
    para_5g[k] = pd.read_csv(para_5g[k][0], encoding='gbk', low_memory=False, usecols=para_5g[k][1])

# 4g数据筛选
para_4g['4g工参'] = para_4g['4g工参'][para_4g['4g工参']['厂家'] == '华为']
para_4g['4g异常数据'] = para_4g['4g异常数据'][para_4g['4g异常数据']['执行结果'] == '报文 : 网元断连!']
para_4g['查询小区QCI参数'] = para_4g['查询小区QCI参数'][para_4g['查询小区QCI参数']['服务质量等级'] == 3]

# 5g数据筛选
para_5g['5g工参'] = para_5g['5g工参'][para_5g['5g工参']['厂家'] == '华为']
para_5g['5g异常数据'] = para_5g['5g异常数据'][para_5g['5g异常数据']['执行结果'] == '报文 : 网元断连!']

# 生成4g基础小区表

# 1、小区基础表=小区静态参数表+工参+小区动态参数表
print('>>>共查询到{}个4G小区'.format(len(para_4g['查询小区静态参数'])))
base_4g = para_4g['查询小区静态参数'].merge(para_4g['4g工参'], sort=False, how='left', left_on=['小区名称'], right_on=['小区英文名'])
base_4g.drop(['小区英文名'], inplace=True, axis=1)
base_4g = base_4g.merge(para_4g['查询小区动态参数'], sort=False, how='left', on=['NAME', '本地小区标识'])

# 2、小区基础表+查询NSA_DC管理参数配置
base_4g = base_4g.merge(para_4g['查询NSA_DC管理参数配置'], sort=False, how='left', on=['NAME', '本地小区标识'])

# 3、小区基础表+查询NR_SCG频点配置
_tmp = para_4g['查询NR_SCG频点配置'].groupby(['NAME'], sort=False, as_index=False)['NSA DC B1事件RSRP门限(毫瓦分贝)'].max()
base_4g = base_4g.merge(_tmp, sort=False, how='left', on='NAME')
del _tmp

# 4、小区基础表+查询主载波频点配置核查两项：频点优先级是否符合要求、非锚点向锚点定向切换是否符合要求
'''
由于查询主载波频点配置这个表不是小区级的，而是基站频点级的，统计每个基站的频点配置不符合要求，则判定该基站不符合要求
所有该基站的三个小区都不符合要求
'''
para_4g['查询主载波频点配置']['频点优先级是否符合要求'] = check_freq_priority(para_4g['查询主载波频点配置']['主载波下行频点'],
                                                          para_4g['查询主载波频点配置']['NSA PCC锚点优先级'])
para_4g['查询主载波频点配置']['频点优先级是否符合要求'].replace(0, '否', inplace=True)
para_4g['查询主载波频点配置']['频点优先级是否符合要求'].replace(1, '是', inplace=True)
_tmp = para_4g['查询主载波频点配置'][['NAME', '频点优先级是否符合要求']].loc[para_4g['查询主载波频点配置']['频点优先级是否符合要求'] == '否']
_tmp.drop_duplicates(keep='first', inplace=True)
base_4g = base_4g.merge(_tmp, sort=False, how='left', on='NAME')
base_4g['频点优先级是否符合要求'].fillna('是', inplace=True)
del _tmp

para_4g['查询主载波频点配置']['非锚点向锚点定向切换是否符合要求'] = para_4g['查询主载波频点配置']['NSA DC主载波A4事件的RSRP触发门限(毫瓦分贝)'].apply(
    lambda x: '是' if x == -105 else '否')
_tmp = para_4g['查询主载波频点配置'][['NAME', '非锚点向锚点定向切换是否符合要求']].loc[para_4g['查询主载波频点配置']['非锚点向锚点定向切换是否符合要求'] == '否']
_tmp.drop_duplicates(keep='first', inplace=True)
base_4g = base_4g.merge(_tmp, sort=False, how='left', on='NAME')
base_4g['非锚点向锚点定向切换是否符合要求'].fillna('是', inplace=True)
del _tmp

# 5、小区基础表+查询小区QCI参数
base_4g = base_4g.merge(para_4g['查询小区QCI参数'], sort=False, how='left', on=['NAME', '本地小区标识'])

# 6、小区基础表+查询异频切换参数组
base_4g = base_4g.merge(para_4g['查询异频切换参数组'], sort=False, how='left', left_on=['NAME', '本地小区标识', 'NSA DC异频切换参数组ID'],
                        right_on=['NAME', '本地小区标识', '异频切换参数组ID'])

# 7、规范化表头
base_4g.rename(columns={'NAME': '基站名称'}, inplace=True)
column1 = list(base_4g.columns)
column2 = [x if x.startswith('LTE') else 'LTE' + x for x in column1]
column3 = dict(zip(column1, column2))
base_4g.rename(columns=column3, inplace=True)

# 生成5g基础表
# 1、5g基础表=查询NR_DU小区静态参数+查询NR小区动态参数
print('>>>共查询到{}个5G小区'.format(len(para_5g['查询NR_DU小区静态参数'])))
para_5g['查询NR小区动态参数'].rename(columns={'NR小区标识': 'NR DU小区标识'}, inplace=True)
base_5g = para_5g['查询NR_DU小区静态参数'].merge(para_5g['查询NR小区动态参数'], sort=False, how='left', on=['NAME', 'NR DU小区标识'])

# 2、5g基础表+查询gNodeB跟踪区域信息：添加tac
base_5g = base_5g.merge(para_5g['查询gNodeB跟踪区域信息'], sort=False, how='left', on='NAME')

# 3、5g基础表+查询gNodeB运营商信息：添加网络架构
base_5g = base_5g.merge(para_5g['查询gNodeB运营商信息'], sort=False, how='left', on='NAME')

# 4、5g基础表+查询gNodeB功能：添加gnodebid
base_5g = base_5g.merge(para_5g['查询gNodeB功能'], sort=False, how='left', on='NAME')

# 5、5g基础表+查询NR小区NSA_DC参数配置
para_5g['查询NR小区NSA_DC参数配置'].rename(columns={'NR小区标识': 'NR DU小区标识'}, inplace=True)
base_5g = base_5g.merge(para_5g['查询NR小区NSA_DC参数配置'], sort=False, how='left', on=['NAME', 'NR DU小区标识'])

# 6、5g基础表+工参
base_5g = base_5g.merge(para_5g['5g工参'], sort=False, how='left', left_on='NR DU小区名称', right_on='小区中文名')

# 7、规范化表头
base_5g.drop(['网管中网元名称', '小区中文名'], inplace=True, axis=1)
base_5g.rename(columns={'NAME': '基站名称', 'NR DU小区名称': '小区名称', 'gNodeB标识': '基站标识'}, inplace=True)
column1 = list(base_5g.columns)
column2 = [x if x.startswith('NR') else 'NR' + x for x in column1]
column3 = dict(zip(column1, column2))
base_5g.rename(columns=column3, inplace=True)
base_5g['NRcgi'] = '460-00-' + base_5g['NR基站标识'].map(str) + '-' + base_5g['NR小区标识'].map(str)

# 45g邻区+4g基础表
print('>>>共查询到{}条45g邻区'.format(len(para_4g['查询NR邻区关系'])))
para_4g['查询NR邻区关系'].rename(columns={'NAME': '基站名称'}, inplace=True)
ncell = para_4g['查询NR邻区关系'].merge(base_4g, how='inner', sort=False, left_on=['基站名称', '本地小区标识'],
                                  right_on=['LTE基站名称', 'LTE本地小区标识'])

# 45g邻区+5g基础表
ncell = ncell.merge(base_5g, how='left', sort=False, left_on=['基站标识', '小区标识'], right_on=['NR基站标识', 'NR小区标识'])

# 1、5G未配置锚点核查
_ncell = para_4g['查询NR邻区关系'].loc[:, ('基站标识', '小区标识', '基站名称')]
_ncell.drop_duplicates(keep='first', inplace=True)
_base_5g = base_5g[(base_5g['NR小区可用状态'] == '可用') & (base_5g['NR架构选项'] == '非独立组网模式')]
_base_5g = _base_5g.merge(_ncell, sort=False, how='left', left_on=['NR基站标识', 'NR小区标识'], right_on=['基站标识', '小区标识'])
result = _base_5g[_base_5g['基站名称'].isnull()]
if len(result) > 0:
    result = result[['NR所属地市', 'NR基站名称', 'NR小区标识', 'NR基站标识', 'NR小区名称', 'NR小区可用状态', 'NR架构选项']]
    result.to_csv(os.path.join(pth, '5G小区未配置锚点.csv'), encoding='gbk', index=False)
    print('>>>5G未配置锚点核查：共核查出{}个5g小区未配置锚点'.format(len(result)))
del _ncell, _base_5g, result

# 2、45g冗余邻区核查
redundance_ncell = ncell[ncell['NR小区名称'].isnull()]
if len(redundance_ncell) > 0:
    # redundance_ncell_need_columns = ['LTE地市', '基站名称', '本地小区标识', 'LTE小区名称', '基站标识', '小区标识']
    redundance_ncell = redundance_ncell[['LTE地市', '基站名称', '本地小区标识', 'LTE小区名称', '基站标识', '小区标识']]
    redundance_ncell.to_csv(os.path.join(pth, '45g冗余邻区.csv'), encoding='gbk', index=False)
    print('>>>45G冗余邻区核查：共有{}条45G邻区未匹配到5G数据'.format(len(redundance_ncell)))
del redundance_ncell

# 3、外部邻区一致性核查
ext_ncell = para_4g['查询NR外部小区'].merge(base_5g, sort=False, how='inner', left_on=['基站标识', '小区标识'],
                                      right_on=['NR基站标识', 'NR小区标识'])
if len(ext_ncell) > 0:
    ext_ncell['ssb频点核查'] = check_ssbfreq(ext_ncell['NRSSB频域位置描述方式'].map(lambda x: 0 if x == '绝对频点' else 1),
                                         ext_ncell['NRSSB频域位置'].map(int), ext_ncell['下行频点'].map(int))
    ext_ncell['ssb频点核查'].replace(0, 'FALSE', inplace=True)
    ext_ncell['ssb频点核查'].replace(1, 'TRUE', inplace=True)
    ext_ncell['ssb频点核查'].replace(2, 'OUTOFRANGE', inplace=True)
    ext_ncell['物理小区标志核查'] = check_same(ext_ncell['物理小区标识'].map(int), ext_ncell['NR物理小区标识'].map(int))
    ext_ncell['物理小区标志核查'].replace(0, 'FALSE', inplace=True)
    ext_ncell['物理小区标志核查'].replace(1, 'TRUE', inplace=True)
    ext_ncell['跟踪区域码核查'] = check_same(ext_ncell['跟踪区域码'].map(int), ext_ncell['NR跟踪区域码'].map(int))
    ext_ncell['跟踪区域码核查'].replace(0, 'FALSE', inplace=True)
    ext_ncell['跟踪区域码核查'].replace(1, 'TRUE', inplace=True)
    ext_ncell = ext_ncell[
        (ext_ncell['物理小区标志核查'] == 'FALSE') | (ext_ncell['跟踪区域码核查'] == 'FALSE') | (ext_ncell['ssb频点核查'] == 'FALSE') | (
                ext_ncell['ssb频点核查'] == 'OUTOFRANGE')]
    if len(ext_ncell):
        need_columns = ['NR所属地市', 'NAME', '基站标识', '小区标识', '下行频点', '物理小区标识', '跟踪区域码',
                        'NRSSB频域位置描述方式', 'NRSSB频域位置', 'NR物理小区标识', 'NR跟踪区域码', 'NR小区可用状态',
                        'ssb频点核查', '物理小区标志核查', '跟踪区域码核查']
        ext_ncell = ext_ncell[need_columns]
        ext_ncell.rename(columns={'NR所属地市': '地市', 'NAME': '基站名称'}, inplace=True)
        ext_ncell.to_csv(os.path.join(pth, '45G外部小区不一致.csv'), encoding='gbk', index=False)
        print('>>>外部邻区一致性核查:共有{}条外部邻区不一致'.format(len(ext_ncell)))
        ext_ncell['外部小区一致性核查'] = '否'
        ext_ncell = ext_ncell[['基站名称', '基站标识', '小区标识', '外部小区一致性核查']]
        ncell = ncell.merge(ext_ncell, sort=False, how='left', on=['基站名称', '基站标识', '小区标识'])
        ncell['外部小区一致性核查'].fillna('是', inplace=True)
    else:
        ncell['外部小区一致性核查'] = '是'
    del ext_ncell

# 4、核查频点优先级是否符合要求、非锚点向锚点定向切换是否符合要求，输出不符合要求的表
output = para_4g['查询主载波频点配置'][
    (para_4g['查询主载波频点配置']['频点优先级是否符合要求'] == '否') | (para_4g['查询主载波频点配置']['非锚点向锚点定向切换是否符合要求'] == '否')]
if len(output):
    _tmp = para_4g['查询NSA_DC管理参数配置'][['NAME', 'NSA DC算法开关&NSA DC能力开关']]
    _tmp = _tmp[_tmp['NSA DC算法开关&NSA DC能力开关'] == '开']
    _tmp.drop_duplicates(keep='first', inplace=True)
    output = output.merge(_tmp, how='left', sort=False, on='NAME')
    output['NSA DC算法开关&NSA DC能力开关'].fillna('关', inplace=True)
    output.to_csv(os.path.join(pth, '查询主载波频点配置问题.csv'), encoding='gbk', index=False, mode='w')
    print('>>>LTE主载波频点配置中共核查出{}条不符合集团规范配置'.format(len(output)))
del output

# 导出4G和5G基础信息表
base_4g.to_csv(os.path.join(pth, '4G基础信息表.csv'), encoding='gbk', index=False, mode='w')
print('>>>导出4G基础数据...')
base_5g.to_csv(os.path.join(pth, '5G基础信息表.csv'), encoding='gbk', index=False, mode='w')
print('>>>导出5G基础数据...')

del para_4g  # 删除变量节省内存
del para_5g
del base_4g
del base_5g

# 5、ncell除去非锚点和匹配不出来5g数据的邻区关系
ncell = ncell[(ncell['NR小区名称'].notnull()) & (ncell['LTENSA DC算法开关&NSA DC能力开关'] == '开')]
print('>>>除去冗余数据和非锚点数据后，共剩余{}条45g邻区关系'.format(len(ncell)))

# 6、ncell规范表头
ncell.rename(columns={'NAME': '基站名称'})

# 7、添加部分缺失字段
ncell['配置锚点小区的数量'] = ncell.groupby('NRcgi', sort=False)['NR小区名称'].transform('count')
ncell['与5G小区的共覆盖比例'] = '100%'
ncell['省份'] = '河北'
ncell['与5G小区的共覆盖比例'] = '100%'
ncell['NR厂家'].fillna('华为', inplace=True)
ncell['LTE厂家'].fillna('华为', inplace=True)
ncell['添加和删除之间的GP'] = ncell['LTENSA DC B1事件RSRP门限(毫瓦分贝)'] - ncell['NRPSCell A2事件RSRP门限(毫瓦分贝)']
ncell['距离'] = getdistance(ncell['LTE纬度'].map(float), ncell['LTE经度'].map(float), ncell['NR纬度'].map(float),
                          ncell['NR经度'].map(float))

# 核查NR小区参数配置是否合规
ncell['是否按照锚点优化原则完成优化'] = ncell.apply(lambda x: '是' if -115 <= x['NRPSCell A2事件RSRP门限(毫瓦分贝)'] <= -105 else '否', axis=1)
print('>>>5G基于覆盖的SN删除门限(PscellA2RsrpThld)共核查出{}条不符合集团标准'.format(len(ncell[ncell['是否按照锚点优化原则完成优化'] == '否'])))


# 核查锚点参数配置是否合规
def check_para(s):
    if s['LTENSA DC B1事件RSRP门限(毫瓦分贝)'] > -100 or s['LTENSA DC B1事件RSRP门限(毫瓦分贝)'] < -105: return '否'
    if s['LTE频点优先级是否符合要求'] == '否': return '否'
    if s['LTE非锚点向锚点定向切换是否符合要求'] == '否': return '否'
    if s['LTENSA DC算法开关&周期性触发PCC锚点选择开关'] == '关': return '否'
    if s['LTE基于A4A5异频A2 RSRP触发门限(毫瓦分贝)'] > -108 or s['LTE基于A4A5异频A2 RSRP触发门限(毫瓦分贝)'] < -108: return '否'
    if s['LTE基于A3的异频A2 RSRP触发门限(毫瓦分贝)'] > -108 or s['LTE基于A3的异频A2 RSRP触发门限(毫瓦分贝)'] < -108: return '否'
    if s['LTE基于覆盖的异频RSRP触发门限(毫瓦分贝)'] > -105 or s['LTE基于覆盖的异频RSRP触发门限(毫瓦分贝)'] < -105: return '否'
    if s['LTE基于A4A5异频A1 RSRP触发门限(毫瓦分贝)'] > -105 or s['LTE基于A4A5异频A1 RSRP触发门限(毫瓦分贝)'] < -105: return '否'
    if s['LTE基于A3的异频A1 RSRP触发门限(毫瓦分贝)'] > -105 or s['LTE基于A3的异频A1 RSRP触发门限(毫瓦分贝)'] < -105: return '否'
    return '是'


ncell['是否按照锚点参数要求进行参数配置'] = ncell.apply(check_para, axis=1)
print('>>>4G共核查出{}条参数条不符合集团标准'.format(len(ncell[ncell['是否按照锚点参数要求进行参数配置'] == '否'])))


# 问题汇总1
def check_error1(s):
    result1 = result2 = None
    if s['是否按照锚点参数要求进行参数配置'] == '否': result1 = '锚点'
    if s['是否按照锚点优化原则完成优化'] == '否': result2 = 'NR'
    if result1 and result2:
        return result1 + '&' + result2
    else:
        return result1 or result2


ncell['问题汇总1'] = ncell.apply(check_error1, axis=1)
print('>>>问题类型1：锚点&NR，共核查出{}条'.format(len(ncell[ncell['问题汇总1'] == '锚点&NR'])))
print('>>>问题类型1：锚点，共核查出{}条'.format(len(ncell[ncell['问题汇总1'] == '锚点'])))
print('>>>问题类型1：NR，共核查出{}条'.format(len(ncell[ncell['问题汇总1'] == 'NR'])))


# 问题汇总2
def check_error2(s):
    if s['距离'] > 2000:
        return '超远配置'
    elif s['距离'] < 50 and s['LTE上层指示开关'] == '关':
        return '50米内未开启ULI'
    elif 800 <= s['距离'] <= 2000 and (s['LTE上层指示开关'] == '开' or s['LTE上层指示开关'] == '基于NR邻区关系广播'):
        return '重点核查ULI开启'


ncell['问题汇总2'] = ncell.apply(check_error2, axis=1)
print('>>>问题类型2：超远配置，共核查出{}条'.format(len(ncell[ncell['问题汇总2'] == '超远配置'])))
print('>>>问题类型2：50米内未开启ULI，共核查出{}条'.format(len(ncell[ncell['问题汇总2'] == '50米内未开启ULI'])))
print('>>>问题类型2：重点核查ULI开启，共核查出{}条'.format(len(ncell[ncell['问题汇总2'] == '重点核查ULI开启'])))

need_columns = ['省份', 'LTE地市', 'NRcgi', 'NR经度', 'NR纬度', 'LTE厂家', '是否按照锚点优化原则完成优化', '配置锚点小区的数量',
                'LTEcgi', 'LTE经度', 'LTE纬度', 'LTE小区名称', 'LTE工作频段', 'LTE厂家', '与5G小区的共覆盖比例',
                '是否按照锚点参数要求进行参数配置', '外部小区一致性核查', 'LTE上层指示开关', 'NRPSCell A2事件RSRP门限(毫瓦分贝)',
                'LTENSA DC B1事件RSRP门限(毫瓦分贝)', '添加和删除之间的GP', 'LTENSA DC算法开关&NSA DC能力开关',
                'LTENSA DC算法开关&周期性触发PCC锚点选择开关', 'LTE频点优先级是否符合要求', 'LTE非锚点向锚点定向切换是否符合要求',
                'LTE基于A4A5异频A2 RSRP触发门限(毫瓦分贝)', 'LTE基于A3的异频A2 RSRP触发门限(毫瓦分贝)',
                'LTE基于覆盖的异频RSRP触发门限(毫瓦分贝)', 'LTE基于A4A5异频A1 RSRP触发门限(毫瓦分贝)',
                'LTE基于A3的异频A1 RSRP触发门限(毫瓦分贝)', '距离', '问题汇总1', '问题汇总2']
ncell = ncell[need_columns]

# 特殊需求：针对张家口高铁场景，全部修改为符合规范
print('>>>特殊修改：张家口高铁小区全部修改为符合规范')
ncell['外部小区一致性核查'] = ncell.apply(lambda x: '是' if x['LTE地市'] == '张家口' else x['外部小区一致性核查'], axis=1)
ncell['是否按照锚点优化原则完成优化'] = ncell.apply(lambda x: '是' if x['LTE地市'] == '张家口' else x['是否按照锚点优化原则完成优化'], axis=1)
ncell['是否按照锚点参数要求进行参数配置'] = ncell.apply(lambda x: '是' if x['LTE地市'] == '张家口' else x['是否按照锚点参数要求进行参数配置'], axis=1)
ncell['问题汇总1'] = ncell.apply(lambda x: None if x['LTE地市'] == '张家口' else x['问题汇总1'], axis=1)
ncell['问题汇总2'] = ncell.apply(lambda x: None if x['LTE地市'] == '张家口' else x['问题汇总2'], axis=1)

ncell.to_csv(os.path.join(pth, '锚点核查最终表.csv'), encoding='gbk', index=False)
print('完成核查：最终核查数量为{}条邻区'.format(len(ncell)))
