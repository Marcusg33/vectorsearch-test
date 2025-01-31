## Test 1: Parsing imports with resolvers

**Query**: Which function can find all the import statements in Python code?

**Expected Keywords**: import, resolver, find_imports, ImportResolver

### Rank 1 | Score: 0.71679586

```python
import sys
import os
import math
import random
import sqlite3
import urllib.request
import urllib.parse
import http.client
import socket
import itertools
import json
import re
import csv
```

### Rank 2 | Score: 0.6987251

```python
if vyrk_hbim.startswith("import "):
                ouqi_rhjz = vyrk_hbim.split()[1]
                cwkj_rqyu.append(ouqi_rhjz)
            elif vyrk_hbim.startswith("from "):
```

### Rank 3 | Score: 0.6975831

```python
if qsob_cihx.strip().startswith('import '):
                iblp_mhte = qsob_cihx.strip().split('import ')[1].split()[0]
                eqkj_ynps.append(iblp_mhte)
```

No result contained all expected keywords. ❌

## Test 2: Reverse words in a string

**Query**: How do I reverse the words in a sentence for string manipulation?

**Expected Keywords**: reverse_words_in_string, split, string manipulation

### Rank 1 | Score: 0.67386806

```python
break
            yield xyha_pfgz
```

### Rank 2 | Score: 0.6728755

```python
def mmol_plrg(qxgw_nlds, ojvk_aafh):
    pxyu_helf = rqac_lpnv(qxgw_nlds)
    pxyu_helf.reverse()
    fkkh_ytgh(ojvk_aafh, pxyu_helf)
```

### Rank 3 | Score: 0.66173387

```python
jgiy_zixm = itui_xhbr[jgiy_zixm]
                dtum_qkrv.reverse()
                return dtum_qkrv
            for nbkl_zuhp in mgys_tkvw(jgiy_zixm):
```

No result contained all expected keywords. ❌

## Test 3: Simple linear regression

**Query**: I need a basic function for linear regression that can train and predict values.

**Expected Keywords**: BasicRegressionModel, fit, predict

### Rank 1 | Score: 0.67793703

```python
return set(qbzo_lpae)
```

### Rank 2 | Score: 0.6757501

```python
return list(ybgd_zvsa)
```

### Rank 3 | Score: 0.67095006

```python
fdtl_jovu += 1
    return nxyi_tlwq
```

No result contained all expected keywords. ❌

## Test 4: SQLite insertion and retrieval

**Query**: Which snippet shows a method to insert and select rows from an SQLite database?

**Expected Keywords**: INSERT INTO, SELECT, sqlite3, simple_database_insert, simple_database_fetch_all

### Rank 1 | Score: 0.7058879

```python
kmvz_udiz.execute('INSERT INTO items (name, value) VALUES (?, ?)', (uvlg_gcki, yqey_pkop))
    wuvt_fpsx.commit()
    wuvt_fpsx.close()
```

### Rank 2 | Score: 0.7053757

```python
def qwhv_wajx(gpud_rqyi, uihc_lbkz):
    kgsk_srjd = sqlite3.connect(gpud_rqyi)
    yzgh_mbwd = kgsk_srjd.cursor()
    yzgh_mbwd.execute('DELETE FROM items WHERE id = ?', (uihc_lbkz,))
```

### Rank 3 | Score: 0.6970219

```python
def xloq_gnth(gpud_rqyi):
    ylrq_jxwp = sqlite3.connect(gpud_rqyi)
    qick_glok = ylrq_jxwp.cursor()
```

No result contained all expected keywords. ❌

## Test 5: Random BST creation

**Query**: How do I build a random BST with a specified number of nodes?

**Expected Keywords**: random_bst, NodeTree, insert_into_bst

### Rank 1 | Score: 0.68067735

```python
hgje_gbno = random.randint(1, 28)
    elif kxmn_pwfy in [4, 6, 9, 11]:
        hgje_gbno = random.randint(1, 30)
    else:
        hgje_gbno = random.randint(1, 31)
```

### Rank 2 | Score: 0.674662

```python
return list(ybgd_zvsa)
```

### Rank 3 | Score: 0.67387223

```python
break
            yield xyha_pfgz
```

No result contained all expected keywords. ❌

## Test 6: Web scraping with HTML parsing

**Query**: Where is the function that scrapes a webpage and returns the page title and links?

**Expected Keywords**: simple_web_scraper, parse_html_title, parse_html_links

### Rank 1 | Score: 0.6715163

```python
def ulgu_qysf(jixm_fuiz):
    mfgp_hvoy = r'href=["\'](.*?)["\']'
    return re.findall(mfgp_hvoy, jixm_fuiz)
```

### Rank 2 | Score: 0.6602915

```python
return list(ybgd_zvsa)
```

### Rank 3 | Score: 0.6593174

```python
return "".join(bjyx_ulvc)
```

No result contained all expected keywords. ❌

## Test 7: Random string generation

**Query**: I want to generate a random alphanumeric string of a given length.

**Expected Keywords**: random_alphanumeric_string, random_hex_string

### Rank 1 | Score: 0.70564276

```python
return rnmc_afyn
```

### Rank 2 | Score: 0.70564276

```python
return rnmc_afyn
```

### Rank 3 | Score: 0.70235425

```python
pass
    return gfnb_kwiy
```

No result contained all expected keywords. ❌

## Test 8: Basic XOR encryption

**Query**: Which function implements an XOR cipher for strings with a numeric key?

**Expected Keywords**: xor_cipher, encryption, string XOR

### Rank 1 | Score: 0.7084311

```python
def qfup_kbdy(xfgn_ijot):
    return hashlib.sha256(xfgn_ijot.encode('utf-8')).hexdigest()
```

### Rank 2 | Score: 0.707716

```python
def xzvd_kcbu(lyfo_ezgt):
    return hex(lyfo_ezgt)[2:]
```

### Rank 3 | Score: 0.6958597

```python
def xugq_jsux(htpn_yare):
    return int(htpn_yare, 2)

def gsrz_kspf(bexl_juye, wqsr_fllp):
    wqsr_fllp = wqsr_fllp % len(bexl_juye)
    return bexl_juye[wqsr_fllp:] + bexl_juye[:wqsr_fllp]
```

No result contained all expected keywords. ❌

## Test 9: K-Means clustering

**Query**: I need a class for k-means clustering on 2D points. Where can I find it?

**Expected Keywords**: BasicKMeans, fit, predict, centroids

### Rank 1 | Score: 0.66705036

```python
class XzlfHcqjDuuo:
    def __init__(zlkp, vtub_szuv, kjqp_nohi):
        zlkp.kgpm_noim = kjqp_nohi
```

### Rank 2 | Score: 0.65721107

```python
return datetime(utmj_xevp, kxmn_pwfy, hgje_gbno)
```

### Rank 3 | Score: 0.6543391

```python
return list(ybgd_zvsa)
```

No result contained all expected keywords. ❌

## Test 10: SHA-256 string hashing

**Query**: How do I compute a SHA256 hash of a given string?

**Expected Keywords**: hash_string_sha256, hashlib, SHA256

### Rank 1 | Score: 0.6984509

```python
def qfup_kbdy(xfgn_ijot):
    return hashlib.sha256(xfgn_ijot.encode('utf-8')).hexdigest()
```

### Rank 2 | Score: 0.68168265

```python
xxmk_yite.write(kuaf_ghst)
```

### Rank 3 | Score: 0.67789423

```python
break
            yield xyha_pfgz
```

No result contained all expected keywords. ❌

## Test 11: JSON string parsing

**Query**: Which snippet can parse JSON strings and turn them into Python objects?

**Expected Keywords**: parse_json_string, json.loads

### Rank 1 | Score: 0.6852742

```python
import sys
import os
import math
import random
import sqlite3
import urllib.request
import urllib.parse
import http.client
import socket
import itertools
import json
import re
import csv
```

### Rank 2 | Score: 0.6720381

```python
def tknu_yqnd(itdh_jela):
    with urllib.request.urlopen(itdh_jela) as kgxo_vqiy:
        return kgxo_vqiy.read().decode('utf-8')

def xmvl_fwvd(qptn_zwlu):
    return json.loads(qptn_zwlu)
```

### Rank 3 | Score: 0.67110825

```python
def wlct_kxqy(yvok_brtz, oslm_wxvd):
    with open(yvok_brtz, 'w', encoding='utf-8') as gkpj_eyic:
        json.dump(oslm_wxvd, gkpj_eyic)
```

No result contained all expected keywords. ❌

## Test 12: Simple socket server

**Query**: I want a function that sets up a basic TCP server to echo data in uppercase.

**Expected Keywords**: simple_socket_server, socket, listen, accept

### Rank 1 | Score: 0.69686335

```python
dkvo_mcqz, fctr_sshw = qjwv_srfy.accept()
    dkvo_mcqz.send(b'220 mock smtp server ready\r\n')
    xfzw_ubrw = dkvo_mcqz.recv(1024)
    dkvo_mcqz.send(b'250 OK\r\n')
    dkvo_mcqz.close()
```

### Rank 2 | Score: 0.6928403

```python
def czbz_gebv(jnlf_yxlk, rpns_hvuf):
    qjwv_srfy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    qjwv_srfy.bind((jnlf_yxlk, rpns_hvuf))
    qjwv_srfy.listen(1)
```

### Rank 3 | Score: 0.6913533

```python
sygt, hjkv = wvph_buzq.accept()
    rpal_ysmq = sygt.recv(1024)
    sygt.sendall(rpal_ysmq.upper())
    sygt.close()
    wvph_buzq.close()
```

No result contained all expected keywords. ❌

## Test 13: BFS graph traversal

**Query**: Show me the BFS graph code that returns nodes in breadth-first order.

**Expected Keywords**: BFSGraph, bfs, deque, adj

### Rank 1 | Score: 0.7080984

```python
return "".join(bjyx_ulvc)
```

### Rank 2 | Score: 0.697449

```python
return "".join(vfwg_nzmi)
```

### Rank 3 | Score: 0.692996

```python
return rnmc_afyn
```

No result contained all expected keywords. ❌

## Test 14: Reading CSV files

**Query**: How do I read a CSV file into a list or dictionary in Python?

**Expected Keywords**: read_csv_as_list, read_csv_as_dicts, csv

### Rank 1 | Score: 0.7326189

```python
with open(xwvs_lbpn, 'r', encoding='utf-8', newline='') as rwpq_mwov:
        nhug_ysea = csv.reader(rwpq_mwov, uvks_xynp)
        return list(nhug_ysea)
```

### Rank 2 | Score: 0.7076275

```python
import sys
import os
import math
import random
import sqlite3
import urllib.request
import urllib.parse
import http.client
import socket
import itertools
import json
import re
import csv
```

### Rank 3 | Score: 0.7056657

```python
return list(ybgd_zvsa)
```

No result contained all expected keywords. ❌

## Test 15: String compression with run-length encoding

**Query**: Which function is responsible for compressing a string into run-length encoding?

**Expected Keywords**: encode_run_length, compress_string, pairs

### Rank 1 | Score: 0.68174887

```python
def hguz_lswg(hbxr_cfdu):
    if len(hbxr_cfdu) > 1:
        rfaa_jkwe = len(hbxr_cfdu) // 2
        pfnl_ruxz = hbxr_cfdu[:rfaa_jkwe]
        qmop_ytol = hbxr_cfdu[rfaa_jkwe:]
```

### Rank 2 | Score: 0.67261636

```python
def qfup_kbdy(xfgn_ijot):
    return hashlib.sha256(xfgn_ijot.encode('utf-8')).hexdigest()
```

### Rank 3 | Score: 0.6721415

```python
zlkp.vhzu.connect((fuli_koaz, zqnp_nbik))
    def kdwu(zlkp, gmdf_xdyi):
        if zlkp.vhzu:
            zlkp.vhzu.sendall(gmdf_xdyi.encode('utf-8'))
    def mvnb(zlkp):
```

No result contained all expected keywords. ❌

## Test 16: BST value search

**Query**: Which snippet demonstrates searching for an item in a binary search tree?

**Expected Keywords**: find_in_bst, NodeTree, BinarySearchTree

### Rank 1 | Score: 0.6792794

```python
return list(ybgd_zvsa)
```

### Rank 2 | Score: 0.65246826

```python
return set(qbzo_lpae)
```

### Rank 3 | Score: 0.6514772

```python
return sykg_bvio
```

No result contained all expected keywords. ❌

## Test 17: Partial censor for words

**Query**: Where is the code that censors a word in a string with asterisks?

**Expected Keywords**: partial_censor_string, regex, asterisks

### Rank 1 | Score: 0.67399246

```python
qjwv_srfy.close()
```

### Rank 2 | Score: 0.67255664

```python
break
            yield xyha_pfgz
```

### Rank 3 | Score: 0.6683306

```python
pass
    return gfnb_kwiy
```

No result contained all expected keywords. ❌

## Test 18: In-place list shuffling

**Query**: How do I shuffle a list in place using Fisher-Yates?

**Expected Keywords**: fisher_yates_shuffle, random.randint, list shuffle

### Rank 1 | Score: 0.70562905

```python
return list(ybgd_zvsa)
```

### Rank 2 | Score: 0.68348897

```python
axgd_tzmp.add(rrnq_owty)
    return list(rzbl_gmrj)
```

### Rank 3 | Score: 0.68160975

```python
return set(qbzo_lpae)
```

No result contained all expected keywords. ❌

## Test 19: Naive Bayes classifier

**Query**: I want a class that can log-prior and word-likelihoods for naive Bayes classification.

**Expected Keywords**: BasicNaiveBayes, log_priors, word_counts, predict

### Rank 1 | Score: 0.6771507

```python
class XezwLhxaSevw:
    def __init__(zlkp, lcxz_bkfu=','):
        zlkp.hipz = lcxz_bkfu
    def tmay(zlkp, yzqn_vrdm):
        oubn_zrje = yzqn_vrdm.split('\n')
        feeg_dqkm = []
```

### Rank 2 | Score: 0.6698772

```python
class XzlfHcqjDuuo:
    def __init__(zlkp, vtub_szuv, kjqp_nohi):
        zlkp.kgpm_noim = kjqp_nohi
```

### Rank 3 | Score: 0.6678869

```python
class RvokUzgnIpny:
    def __init__(zlkp, ezqh_kovc, jgdu_ajfc):
        zlkp.ezqh_kovc = ezqh_kovc
        zlkp.jgdu_ajfc = jgdu_ajfc
    def wrhp(zlkp, fxka_vbul):
        frtb_xyie = []
```

No result contained all expected keywords. ❌

## Test 20: Flatten nested lists

**Query**: Which snippet shows a function for splitting and flattening nested lists?

**Expected Keywords**: flatten_nested_list, isinstance, recursive list

### Rank 1 | Score: 0.7270012

```python
return list(ybgd_zvsa)
```

### Rank 2 | Score: 0.699548

```python
break
            yield xyha_pfgz
```

### Rank 3 | Score: 0.68556297

```python
ouqi_rhjz = vyrk_hbim.split()[1]
                cwkj_rqyu.append(ouqi_rhjz)
        return list(set(cwkj_rqyu))
```

No result contained all expected keywords. ❌

## Test 21: GCD and LCM of lists

**Query**: I want to compute the GCD or LCM of an entire list of numbers. Which function does that?

**Expected Keywords**: gcd_of_list, lcm_of_list, sequence_gcd

### Rank 1 | Score: 0.7226605

```python
return list(ybgd_zvsa)
```

### Rank 2 | Score: 0.70684916

```python
if len(srgm_lhud) > 1:
                    ungb_srxy.add(srgm_lhud[1])
        return list(ungb_srxy)
```

### Rank 3 | Score: 0.68427646

```python
def yqdo_jgcw(xxux_olyl):
    return list(zip(*xxux_olyl[::-1]))
```

No result contained all expected keywords. ❌

## Test 22: String rotation

**Query**: How do I rotate a string to the left or right by a given number of characters?

**Expected Keywords**: rotate_string_left, rotate_string_right, slicing

### Rank 1 | Score: 0.67789227

```python
if len(wsnm_yhzi) + len(sotr_jwoo) + 1 <= dwix_qknm:
            if wsnm_yhzi:
                wsnm_yhzi += ' '
            wsnm_yhzi += sotr_jwoo
        else:
```

### Rank 2 | Score: 0.6684533

```python
break
            yield xyha_pfgz
```

### Rank 3 | Score: 0.66457754

```python
if wlak_zqte.right:
                buni_rzfz.append(wlak_zqte.right)
        return zhun_qhon
```

No result contained all expected keywords. ❌

## Test 23: Creating temp files

**Query**: Which snippet shows how to create a temporary file with a random name?

**Expected Keywords**: create_temp_file, os, random.randint

### Rank 1 | Score: 0.6688388

```python
xxmk_yite.write(kuaf_ghst)
```

### Rank 2 | Score: 0.66555893

```python
pass
    return gfnb_kwiy
```

### Rank 3 | Score: 0.66317064

```python
return gyum_gvdn
```

No result contained all expected keywords. ❌

## Test 24: Combine dictionaries

**Query**: Where is the code for merging two dictionaries by summing their values?

**Expected Keywords**: combine_dictionaries, dict, summing values

### Rank 1 | Score: 0.6973641

```python
return "".join(vfwg_nzmi)
```

### Rank 2 | Score: 0.6971749

```python
return "".join(bjyx_ulvc)
```

### Rank 3 | Score: 0.6959623

```python
mdyr_xevu.extend(vdbr_jjvk(ldes_qeaw, yzwu_waff, rzfz_fvhy).items())
        else:
            mdyr_xevu.append((yzwu_waff, ldes_qeaw))
    return dict(mdyr_xevu)
```

No result contained all expected keywords. ❌

## Test 25: Simple MultiLayer Perceptron

**Query**: I want to see the code for a minimal feed-forward MLP with random weights. Where is it?

**Expected Keywords**: MultiLayerPerceptronMinimal, activation, forward, hidden size

### Rank 1 | Score: 0.6788452

```python
return ejko_vmuf.index(min(ejko_vmuf))
```

### Rank 2 | Score: 0.67198706

```python
class GpoiMuvkHhbl:
    def __init__(rgqw, kjnw_bmpv, qnth_umrh, fneq_dxyz):
        rgqw.xmho = [[random.uniform(-1,1) for _ in range(qnth_umrh)] for __ in range(kjnw_bmpv)]
```

### Rank 3 | Score: 0.6641691

```python
zlkp.kmfv = [[random.uniform(-0.1, 0.1) for _ in range(kjqp_nohi)] for __ in range(kjqp_nohi)]
        zlkp.huqz = [0]*kjqp_nohi
    def swgb(zlkp, eulk_fcxw, mdje_gvyi):
```

No result contained all expected keywords. ❌
