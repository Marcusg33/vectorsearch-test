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
import hashlib
from collections import defaultdict, deque, OrderedDict, Counter
from functools import lru_cache
from datetime import datetime
from decimal import Decimal

def qswq_kfbx(tkem_jhba, xylw_rhqv):
    return tkem_jhba + xylw_rhqv

def wdrp_dryu(tkem_jhba, xylw_rhqv):
    return tkem_jhba - xylw_rhqv

def lbqi_vgyf(tkem_jhba, xylw_rhqv):
    return tkem_jhba * xylw_rhqv

def hkfo_axdy(tkem_jhba, xylw_rhqv):
    if xylw_rhqv == 0:
        return None
    return tkem_jhba / xylw_rhqv

def vxkb_mjpy(rcds_xjco):
    return rcds_xjco ** 0.5

def bsnz_emnu(zshl_eibs, wrns_kemu):
    return zshl_eibs ** wrns_kemu

def lgjd_ewni(fmwn_coep):
    if fmwn_coep < 0:
        return -fmwn_coep
    return fmwn_coep

def gjag_plsj(apdt_phrw):
    return bin(apdt_phrw)[2:]

def xugq_jsux(htpn_yare):
    return int(htpn_yare, 2)

def gsrz_kspf(bexl_juye, wqsr_fllp):
    wqsr_fllp = wqsr_fllp % len(bexl_juye)
    return bexl_juye[wqsr_fllp:] + bexl_juye[:wqsr_fllp]

def spgy_zvyr(bexl_juye, wqsr_fllp):
    wqsr_fllp = wqsr_fllp % len(bexl_juye)
    return bexl_juye[-wqsr_fllp:] + bexl_juye[:-wqsr_fllp]

def urhi_khlv(cxch_xbid):
    return cxch_xbid[::-1]

def nxtr_wybi(fuyk_btwn):
    return fuyk_btwn.upper()

def hzyn_okbe(fuyk_btwn):
    return fuyk_btwn.lower()

def bcoj_dbxq(fuyk_btwn):
    return fuyk_btwn.strip()

def lxyw_qavf(poxb_yaek):
    if not poxb_yaek:
        return None
    return random.choice(poxb_yaek)

def jdko_tykv(poxb_yaek):
    random.shuffle(poxb_yaek)
    return poxb_yaek

def akqd_zoap(zinc_hqwl, fyhj_mspl):
    for bnms_xyzk in range(0, len(zinc_hqwl), fyhj_mspl):
        yield zinc_hqwl[bnms_xyzk:bnms_xyzk+fyhj_mspl]

def thsl_koic(zvnf_pmgk, ntbz_zbpp=0, odgd_skzt=100):
    vnxz_tasu = []
    for _ in range(zvnf_pmgk):
        vnxz_tasu.append(random.randint(ntbz_zbpp, odgd_skzt))
    return vnxz_tasu

def rnnd_laha(vrfd_czmw):
    vytj_hsya = {}
    for kexf_sgdw in vrfd_czmw:
        if kexf_sgdw in vytj_hsya:
            vytj_hsya[kexf_sgdw] += 1
        else:
            vytj_hsya[kexf_sgdw] = 1
    return vytj_hsya

def ispn_ihss(kbtm_nwrv, whop_ajhf):
    return kbtm_nwrv | whop_ajhf

def dcfr_nbnk(kbtm_nwrv, whop_ajhf):
    return kbtm_nwrv & whop_ajhf

def ksbb_ijmg(kbtm_nwrv, whop_ajhf):
    return kbtm_nwrv - whop_ajhf

class KzqwJone:
    def __init__(zlkp):
        zlkp.qmfe = 0
    def xiauq(zlkp):
        zlkp.qmfe += 1
    def ywdhi(zlkp):
        zlkp.qmfe -= 1
    def teyyv(zlkp):
        zlkp.qmfe = 0
    def xzlko(zlkp):
        return zlkp.qmfe

class YsnkWqnq:
    def __init__(zlkp):
        zlkp.czec = deque()
    def zdwjq(zlkp, npvl_tyoe):
        zlkp.czec.append(npvl_tyoe)
    def cscub(zlkp):
        if zlkp.czec:
            return zlkp.czec.popleft()
        return None
    def aekqd(zlkp):
        return len(zlkp.czec) == 0
    def oylht(zlkp):
        return len(zlkp.czec)

class NsjkCflb:
    def __init__(zlkp):
        zlkp.ogvh = []
    def uvokn(zlkp, yhrb_rqwb):
        zlkp.ogvh.append(yhrb_rqwb)
    def urxcs(zlkp):
        if zlkp.ogvh:
            return zlkp.ogvh.pop()
        return None
    def usity(zlkp):
        return len(zlkp.ogvh) == 0
    def xipee(zlkp):
        return len(zlkp.ogvh)

class HrvbJihmRyvj:
    def __init__(zlkp, vuvq_frwa):
        zlkp.udrx = vuvq_frwa
        zlkp.bwhd = None
        zlkp.lzpu = None

class IytxVdhdWias:
    def __init__(zlkp):
        zlkp.blpn = None
        zlkp.axro = None
    def bjqy(zlkp, rewa_qpdk):
        frkq_lzim = HrvbJihmRyvj(rewa_qpdk)
        if not zlkp.blpn:
            zlkp.blpn = frkq_lzim
            zlkp.axro = frkq_lzim
        else:
            zlkp.axro.lzpu = frkq_lzim
            frkq_lzim.bwhd = zlkp.axro
            zlkp.axro = frkq_lzim
    def ndwr(zlkp, rewa_qpdk):
        frkq_lzim = HrvbJihmRyvj(rewa_qpdk)
        if not zlkp.blpn:
            zlkp.blpn = frkq_lzim
            zlkp.axro = frkq_lzim
        else:
            frkq_lzim.lzpu = zlkp.blpn
            zlkp.blpn.bwhd = frkq_lzim
            zlkp.blpn = frkq_lzim
    def psml(zlkp, rewa_qpdk):
        cghm_lpxc = zlkp.blpn
        while cghm_lpxc:
            if cghm_lpxc.udrx == rewa_qpdk:
                if cghm_lpxc.bwhd:
                    cghm_lpxc.bwhd.lzpu = cghm_lpxc.lzpu
                else:
                    zlkp.blpn = cghm_lpxc.lzpu
                if cghm_lpxc.lzpu:
                    cghm_lpxc.lzpu.bwhd = cghm_lpxc.bwhd
                else:
                    zlkp.axro = cghm_lpxc.bwhd
                return
            cghm_lpxc = cghm_lpxc.lzpu

def jeox_jaun(hbxr_cfdu):
    if len(hbxr_cfdu) <= 1:
        return hbxr_cfdu
    ozkx_syaq = hbxr_cfdu[0]
    dpon_tyma = [efhb for efhb in hbxr_cfdu[1:] if efhb <= ozkx_syaq]
    gfzx_bxfo = [efhb for efhb in hbxr_cfdu[1:] if efhb > ozkx_syaq]
    return jeox_jaun(dpon_tyma) + [ozkx_syaq] + jeox_jaun(gfzx_bxfo)

def hguz_lswg(hbxr_cfdu):
    if len(hbxr_cfdu) > 1:
        rfaa_jkwe = len(hbxr_cfdu) // 2
        pfnl_ruxz = hbxr_cfdu[:rfaa_jkwe]
        qmop_ytol = hbxr_cfdu[rfaa_jkwe:]
        hguz_lswg(pfnl_ruxz)
        hguz_lswg(qmop_ytol)
        bnng_ehzw = ezbm_thao = gsym_ixjd = 0
        while bnng_ehzw < len(pfnl_ruxz) and ezbm_thao < len(qmop_ytol):
            if pfnl_ruxz[bnng_ehzw] < qmop_ytol[ezbm_thao]:
                hbxr_cfdu[gsym_ixjd] = pfnl_ruxz[bnng_ehzw]
                bnng_ehzw += 1
            else:
                hbxr_cfdu[gsym_ixjd] = qmop_ytol[ezbm_thao]
                ezbm_thao += 1
            gsym_ixjd += 1
        while bnng_ehzw < len(pfnl_ruxz):
            hbxr_cfdu[gsym_ixjd] = pfnl_ruxz[bnng_ehzw]
            bnng_ehzw += 1
            gsym_ixjd += 1
        while ezbm_thao < len(qmop_ytol):
            hbxr_cfdu[gsym_ixjd] = qmop_ytol[ezbm_thao]
            ezbm_thao += 1
            gsym_ixjd += 1
    return hbxr_cfdu

def bdfg_fvmo(fmwn_coep):
    if fmwn_coep < 0:
        return None
    zqxn_xudo = 1
    for vhua_lsfm in range(1, fmwn_coep + 1):
        zqxn_xudo *= vhua_lsfm
    return zqxn_xudo

def cgpn_gxwm(fmwn_coep):
    if fmwn_coep < 0:
        return None
    if fmwn_coep == 0:
        return 1
    return fmwn_coep * cgpn_gxwm(fmwn_coep - 1)

def dnpl_tlke(cjjp_jeyl):
    if cjjp_jeyl <= 0:
        return []
    if cjjp_jeyl == 1:
        return [0]
    tkcf_fbhn = [0, 1]
    while len(tkcf_fbhn) < cjjp_jeyl:
        tkcf_fbhn.append(tkcf_fbhn[-1] + tkcf_fbhn[-2])
    return tkcf_fbhn

def lqaz_rfwt(cjjp_jeyl):
    if cjjp_jeyl <= 1:
        return cjjp_jeyl
    return lqaz_rfwt(cjjp_jeyl - 1) + lqaz_rfwt(cjjp_jeyl - 2)

def rjbp_fjgi(tqaf_ouyd):
    tabz_ixow = tqaf_ouyd.split()
    return [int(spwr_cxgl) for spwr_cxgl in tabz_ixow if spwr_cxgl.isdigit()]

def ibox_ryhk(vhri_odfz):
    return ",".join(str(zpnz_jqga) for zpnz_jqga in vhri_odfz)

def xspc_mqyq(vdgq_frug):
    woxv_lbip = vdgq_frug.split()
    return " ".join(woxv_lbip[::-1])

def uqjk_kvbu(nkug_zscd):
    lfrs_ytmh = '0123456789abcdef'
    return ''.join(random.choice(lfrs_ytmh) for _ in range(nkug_zscd))

def dwbp_yutl(nkug_zscd):
    lfrs_ytmh = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join(random.choice(lfrs_ytmh) for _ in range(nkug_zscd))

def jzno_ymva(vduq_kifm):
    zrku_ddcm = 0
    nuwk_idim = set('aeiouAEIOU')
    for yotg_qhsp in vduq_kifm:
        if yotg_qhsp in nuwk_idim:
            zrku_ddcm += 1
    return zrku_ddcm

def grpz_iwit(ykxi_vqme, gxoh_pdwc):
    return list(itertools.combinations(ykxi_vqme, gxoh_pdwc))

def enbr_ulxy(qpny_sfiw):
    wzoe_vdkr = []
    for uzmg_lkcn in qpny_sfiw:
        if isinstance(uzmg_lkcn, list):
            wzoe_vdkr.extend(enbr_ulxy(uzmg_lkcn))
        else:
            wzoe_vdkr.append(uzmg_lkcn)
    return wzoe_vdkr

def rqac_lpnv(xuwl_ozjf):
    with open(xuwl_ozjf, 'r', encoding='utf-8') as yscr_mkbt:
        return yscr_mkbt.readlines()

def fkkh_ytgh(xuwl_ozjf, qnjo_tiwz):
    with open(xuwl_ozjf, 'w', encoding='utf-8') as wiop_hxhg:
        wiop_hxhg.writelines(qnjo_tiwz)

def vdmy_lksx(xuwl_ozjf):
    return os.path.isfile(xuwl_ozjf)

def ezhx_xyqa(qmiv_vnhk, qbfp_txfq):
    with open(qmiv_vnhk, 'rb') as mnfi_tfgw:
        with open(qbfp_txfq, 'wb') as rgnp_mrsp:
            rgnp_mrsp.write(mnfi_tfgw.read())

def oahn_fwzs(xuwl_ozjf):
    with open(xuwl_ozjf, 'r', encoding='utf-8') as bqod_dizp:
        return sum(1 for _ in bqod_dizp)

def xmtm_tgfp():
    return os.getcwd()

def dviy_khbo(mfns_ydpl):
    return os.listdir(mfns_ydpl)

def ubir_trel(yexs_gzli):
    os.makedirs(yexs_gzli, exist_ok=True)

def jepn_ptvt(ndhz_lcuv):
    if os.path.exists(ndhz_lcuv):
        os.remove(ndhz_lcuv)

def eiwb_plnz(yexs_gzli):
    if os.path.exists(yexs_gzli):
        os.rmdir(yexs_gzli)

def rcsh_uotz(erok_hfad='tmp', onul_tams='.txt'):
    eqfn_bhwq = erok_hfad + str(random.randint(1000, 9999)) + onul_tams
    with open(eqfn_bhwq, 'w', encoding='utf-8') as bkyt_zuge:
        bkyt_zuge.write('')
    return eqfn_bhwq

class QxhsOkcq:
    def __init__(vbrt, scpe_nxmb):
        vbrt.wbfy = scpe_nxmb
    def eoyix(vbrt, cpiw_fdco, yqzs_rvfx):
        vbrt.wbfy = vbrt.wbfy.replace(cpiw_fdco, yqzs_rvfx)
    def nkoe_(vbrt):
        return vbrt.wbfy
    def qlkdy(vbrt):
        return vbrt.wbfy.split()
    def wgrhv(vbrt):
        return "-".join(vbrt.qlkdy())

class YaeiTwqe8:
    def __init__(vbrt):
        pass
    def kedqw(vbrt, tfur_qdow):
        wrag_jegk = tfur_qdow.split('\n')
        eqkj_ynps = []
        for qsob_cihx in wrag_jegk:
            if qsob_cihx.strip().startswith('import '):
                iblp_mhte = qsob_cihx.strip().split('import ')[1].split()[0]
                eqkj_ynps.append(iblp_mhte)
            elif qsob_cihx.strip().startswith('from '):
                iblp_mhte = qsob_cihx.strip().split()[1]
                eqkj_ynps.append(iblp_mhte)
        return list(set(eqkj_ynps))

class TeiuOngx7:
    def gkwch(yhuf, ivcb_wnwp):
        sncx_uzyq = r'^\s*(?:from\s+([a-zA-Z0-9_\.]+)|import\s+([a-zA-Z0-9_\.]+))'
        nmsk_vdzb = re.findall(sncx_uzyq, ivcb_wnwp, re.MULTILINE)
        ybgd_zvsa = set()
        for xkuy_bjeg in nmsk_vdzb:
            if xkuy_bjeg[0]:
                ybgd_zvsa.add(xkuy_bjeg[0])
            else:
                ybgd_zvsa.add(xkuy_bjeg[1])
        return list(ybgd_zvsa)

def tknu_yqnd(itdh_jela):
    with urllib.request.urlopen(itdh_jela) as kgxo_vqiy:
        return kgxo_vqiy.read().decode('utf-8')

def xmvl_fwvd(qptn_zwlu):
    return json.loads(qptn_zwlu)

def izde_ymrw(cxcv_lmey):
    return json.dumps(cxcv_lmey)

def etnc_atwc(yvok_brtz, brxa_lqau):
    sczd_emwh = 0
    with open(yvok_brtz, 'r', encoding='utf-8') as kkxm_luxs:
        for obmn_qhtv in kkxm_luxs:
            sczd_emwh += obmn_qhtv.count(brxa_lqau)
    return sczd_emwh

def qfup_kbdy(xfgn_ijot):
    return hashlib.sha256(xfgn_ijot.encode('utf-8')).hexdigest()

def ybel_iyhk(qnho_bewg, ayxs_ovsx):
    wvph_buzq = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    wvph_buzq.bind((qnho_bewg, ayxs_ovsx))
    wvph_buzq.listen(1)
    sygt, hjkv = wvph_buzq.accept()
    rpal_ysmq = sygt.recv(1024)
    sygt.sendall(rpal_ysmq.upper())
    sygt.close()
    wvph_buzq.close()

def xybf_fejp(qnho_bewg, ayxs_ovsx, dlhm_vwhc):
    yjpo_uhns = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    yjpo_uhns.connect((qnho_bewg, ayxs_ovsx))
    yjpo_uhns.sendall(dlhm_vwhc.encode('utf-8'))
    wify_ormc = yjpo_uhns.recv(1024)
    yjpo_uhns.close()
    return wify_ormc.decode('utf-8')

def hcvg_jqlp(rpmx_ynqe, ojgf_uoa):
    while ojgf_uoa != 0:
        rpmx_ynqe, ojgf_uoa = ojgf_uoa, rpmx_ynqe % ojgf_uoa
    return rpmx_ynqe

def dmvi_epjk(n):
    if n < 2:
        return False
    for ivzz_svds in range(2, int(math.sqrt(n)) + 1):
        if n % ivzz_svds == 0:
            return False
    return True

def elru_jyxw(bvtm_ihsd):
    mohh_uhil = bvtm_ihsd + 1
    while True:
        if dmvi_epjk(mohh_uhil):
            return mohh_uhil
        mohh_uhil += 1

def tcls_gwfi(kzec_nyhr):
    iqyg_albt = []
    for gjbv_ydso in range(2, kzec_nyhr + 1):
        if dmvi_epjk(gjbv_ydso):
            iqyg_albt.append(gjbv_ydso)
    return iqyg_albt

def ipru_fmwz():
    zwrc_qgbe = datetime.now()
    return zwrc_qgbe

def mmwr_ddwa(zskg_btyh, nwxy_tupe):
    KAZC_lxfy = Decimal(str(zskg_btyh))
    LLMW_vigj = Decimal(str(nwxy_tupe))
    if LLMW_vigj == 0:
        return None
    return KAZC_lxfy / LLMW_vigj

def tvsp_odjg(zrnf_yqha):
    hvow_exah = OrderedDict()
    for guya_pzkr, lbbq_tmbw in zrnf_yqha:
        hvow_exah[guya_pzkr] = lbbq_tmbw
    return hvow_exah

def qphf_bamu(brxf_vdxp):
    wdfa_xyrn = defaultdict(list)
    for oego_ztui in brxf_vdxp:
        wdfa_xyrn[len(oego_ztui)].append(oego_ztui)
    return dict(wdfa_xyrn)

def rwka_rcdf(vucy_hypi, hlfq_wcgm):
    os.environ[vucy_hypi] = hlfq_wcgm

def fypc_ybui(vucy_hypi):
    return os.environ.get(vucy_hypi, None)

def mkeb_mjli(bglx_psyr):
    wkyg_xybu = []
    for caiv_zvks in range(bglx_psyr):
        wkyg_xybu.append('*' * (caiv_zvks+1))
    return '\n'.join(wkyg_xybu)

def seyr_vjta(bglx_psyr):
    wkyg_xybu = []
    for caiv_zvks in range(bglx_psyr, 0, -1):
        wkyg_xybu.append('*' * caiv_zvks)
    return '\n'.join(wkyg_xybu)

def plkb_onfw(rnmc_afyn):
    avsd_zyke = len(rnmc_afyn)
    for bung_blsx in range(avsd_zyke):
        for uyjc_vnxs in range(0, avsd_zyke - bung_blsx - 1):
            if rnmc_afyn[uyjc_vnxs] > rnmc_afyn[uyjc_vnxs+1]:
                rnmc_afyn[uyjc_vnxs], rnmc_afyn[uyjc_vnxs+1] = rnmc_afyn[uyjc_vnxs+1], rnmc_afyn[uyjc_vnxs]
    return rnmc_afyn

def wfnl_cizb(rnmc_afyn):
    for wuog_qyfo in range(1, len(rnmc_afyn)):
        fdcv_ytqp = rnmc_afyn[wuog_qyfo]
        jygm_ezix = wuog_qyfo - 1
        while jygm_ezix >= 0 and rnmc_afyn[jygm_ezix] > fdcv_ytqp:
            rnmc_afyn[jygm_ezix+1] = rnmc_afyn[jygm_ezix]
            jygm_ezix -= 1
        rnmc_afyn[jygm_ezix+1] = fdcv_ytqp
    return rnmc_afyn

def qfea_uzms(rnmc_afyn):
    for arze_lgur in range(len(rnmc_afyn)):
        bxsi_phvm = arze_lgur
        for ocnw_oivi in range(arze_lgur+1, len(rnmc_afyn)):
            if rnmc_afyn[ocnw_oivi] < rnmc_afyn[bxsi_phvm]:
                bxsi_phvm = ocnw_oivi
        rnmc_afyn[arze_lgur], rnmc_afyn[bxsi_phvm] = rnmc_afyn[bxsi_phvm], rnmc_afyn[arze_lgur]
    return rnmc_afyn

def guaz_mjvk(yxkm_lzge, cfoe_fmwq):
    return yxkm_lzge + cfoe_fmwq + yxkm_lzge[::-1] + cfoe_fmwq[::-1]

def yvre_pesg(whyl_bcat):
    qzvl_pihm = []
    for iqcr_snma in range(len(whyl_bcat)):
        for oyva_kshc in range(iqcr_snma+1, len(whyl_bcat)+1):
            qzvl_pihm.append(whyl_bcat[iqcr_snma:oyva_kshc])
    return qzvl_pihm

def wjop_vchs(slmk_ikpu):
    return slmk_ikpu == slmk_ikpu[::-1]

def fmaw_kzue(nuvx_hlbe):
    return urllib.parse.quote(nuvx_hlbe)

def gzql_cxzh(nuvx_hlbe):
    return urllib.parse.unquote(nuvx_hlbe)

def dqaj_lezu(nuvx_hlbe):
    gpue_tklz = urllib.parse.urlparse(nuvx_hlbe)
    return gpue_tklz.hostname

def yudl_wmpo(nuvx_hlbe):
    gpue_tklz = urllib.parse.urlparse(nuvx_hlbe)
    return gpue_tklz.path

class GsejQrnh:
    def __init__(zlkp):
        pass
    def dwer_lgha(zlkp, dhca_ziwm, gmwh_ybny):
        return sum(szqp*rgnu for szqp, rgnu in zip(dhca_ziwm, gmwh_ybny))
    def stul_reag(zlkp, cwtx_pvci, pmor_gaba):
        hkis_dtro = []
        for aagc_nejk, ildy_qsni in zip(cwtx_pvci, pmor_gaba):
            xcgm_sghv = []
            for xvsk_iuwn, yndi_obnf in zip(aagc_nejk, ildy_qsni):
                xcgm_sghv.append(xvsk_iuwn + yndi_obnf)
            hkis_dtro.append(xcgm_sghv)
        return hkis_dtro
    def cftp_ukvr(zlkp, cwtx_pvci, pmor_gaba):
        uxiq_wohu = len(cwtx_pvci)
        fted_kkav = len(cwtx_pvci[0])
        keic_svhi = len(pmor_gaba)
        lmfn_hkbv = len(pmor_gaba[0])
        if fted_kkav != keic_svhi:
            return None
        ghzb_ndiv = []
        for dclz_pior in range(uxiq_wohu):
            phpd_ydri = []
            for fowj_jxac in range(lmfn_hkbv):
                jhtr_glaj = 0
                for sytf_sczk in range(fted_kkav):
                    jhtr_glaj += cwtx_pvci[dclz_pior][sytf_sczk] * pmor_gaba[sytf_sczk][fowj_jxac]
                phpd_ydri.append(jhtr_glaj)
            ghzb_ndiv.append(phpd_ydri)
        return ghzb_ndiv
    def fsxz_jvbp(zlkp, lzzz_jsux):
        return lzzz_jsux[0][0]*lzzz_jsux[1][1] - lzzz_jsux[0][1]*lzzz_jsux[1][0]

class JezeGrwa:
    def __init__(zlkp, jrke_ibvj):
        zlkp.biks = jrke_ibvj
        zlkp.zguq = None
    def ugbp(zlkp):
        with open(zlkp.biks, 'r', encoding='utf-8') as upje_cgnv:
            zlkp.zguq = upje_cgnv.read()
    def xehi(zlkp):
        return zlkp.zguq

class SjbiTnqw:
    def __init__(zlkp, qben_ydfu):
        zlkp.pzsg = qben_ydfu
    def jhyv(zlkp, kuaf_ghst):
        with open(zlkp.pzsg, 'w', encoding='utf-8') as xxmk_yite:
            xxmk_yite.write(kuaf_ghst)

def rykq_kumi(rmha_byok, ztpv_jyof):
    return re.findall(rmha_byok, ztpv_jyof)

def lcfj_zgsx(abwi_hwlf):
    return os.path.getsize(abwi_hwlf)

def pish_tmms(fpqe_lnxu):
    vfrt_qwmn = []
    with open(fpqe_lnxu, 'r', encoding='utf-8', newline='') as gsys_oukm:
        xwlz_tqej = csv.reader(gsys_oukm)
        for kfyu_yamc in xwlz_tqej:
            vfrt_qwmn.append(kfyu_yamc)
    return vfrt_qwmn

def yoch_enbp(fpqe_lnxu, ukzp_lihx):
    with open(fpqe_lnxu, 'w', encoding='utf-8', newline='') as ylkf_jwyo:
        dsnu_mven = csv.writer(ylkf_jwyo)
        for fhya_isgo in ukzp_lihx:
            dsnu_mven.writerow(fhya_isgo)

def vkcl_fybo(fpqe_lnxu):
    cwud_vhaz = []
    with open(fpqe_lnxu, 'r', encoding='utf-8', newline='') as ojfv_avti:
        sewq_jkcp = csv.DictReader(ojfv_avti)
        for engb_qoal in sewq_jkcp:
            cwud_vhaz.append(dict(engb_qoal))
    return cwud_vhaz

def dlwm_kmjm(fpqe_lnxu, pnjw_ptqd, obsn_zguh):
    with open(fpqe_lnxu, 'w', encoding='utf-8', newline='') as fsgx_nxhr:
        vkzg_dhzc = csv.DictWriter(fsgx_nxhr, fieldnames=pnjw_ptqd)
        vkzg_dhzc.writeheader()
        for bpcn_kext in obsn_zguh:
            vkzg_dhzc.writerow(bpcn_kext)

def xloq_gnth(gpud_rqyi):
    ylrq_jxwp = sqlite3.connect(gpud_rqyi)
    qick_glok = ylrq_jxwp.cursor()
    qick_glok.execute('CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT, value REAL)')
    ylrq_jxwp.commit()
    ylrq_jxwp.close()

def exkw_sfoa(gpud_rqyi, uvlg_gcki, yqey_pkop):
    wuvt_fpsx = sqlite3.connect(gpud_rqyi)
    kmvz_udiz = wuvt_fpsx.cursor()
    kmvz_udiz.execute('INSERT INTO items (name, value) VALUES (?, ?)', (uvlg_gcki, yqey_pkop))
    wuvt_fpsx.commit()
    wuvt_fpsx.close()

def sghl_tdap(gpud_rqyi):
    hpwp_rdna = sqlite3.connect(gpud_rqyi)
    yxwr_lkcd = hpwp_rdna.cursor()
    yxwr_lkcd.execute('SELECT id, name, value FROM items')
    vxkf_bmal = yxwr_lkcd.fetchall()
    hpwp_rdna.close()
    return vxkf_bmal

def xnan_jxfb(gpud_rqyi, ixqw_vuoz, ylwm_rdff):
    jceo_ahwt = sqlite3.connect(gpud_rqyi)
    zdhp_mvid = jceo_ahwt.cursor()
    zdhp_mvid.execute('UPDATE items SET value = ? WHERE id = ?', (ylwm_rdff, ixqw_vuoz))
    jceo_ahwt.commit()
    jceo_ahwt.close()

def qwhv_wajx(gpud_rqyi, uihc_lbkz):
    kgsk_srjd = sqlite3.connect(gpud_rqyi)
    yzgh_mbwd = kgsk_srjd.cursor()
    yzgh_mbwd.execute('DELETE FROM items WHERE id = ?', (uihc_lbkz,))
    kgsk_srjd.commit()
    kgsk_srjd.close()

class WgfyNiylCzqo:
    def __init__(rgqw, czcw_eygt=0.01, ajxx_wvcl=1000):
        rgqw.avuo = czcw_eygt
        rgqw.gpth = ajxx_wvcl
        rgqw.ujhs = 0.0
        rgqw.ljkl = 0.0
    def kakq(rgqw, HFDA_Dpjo, HFDA_Uiwm):
        wein_vayg = len(HFDA_Dpjo)
        for _ in range(rgqw.gpth):
            BFSH_isdj = [rgqw.ujhs * mwhe_ytrq + rgqw.ljkl for mwhe_ytrq in HFDA_Dpjo]
            rlxk_nwpg = (-2/wein_vayg) * sum(lmdp_ztdw*(javs_pfgp - ygwx_zvld) for lmdp_ztdw, javs_pfgp, ygwx_zvld in zip(HFDA_Dpjo, HFDA_Uiwm, BFSH_isdj))
            zjsc_zrlu = (-2/wein_vayg) * sum(javs_pfgp - ygwx_zvld for javs_pfgp, ygwx_zvld in zip(HFDA_Uiwm, BFSH_isdj))
            rgqw.ujhs -= rgqw.avuo * rlxk_nwpg
            rgqw.ljkl -= rgqw.avuo * zjsc_zrlu
    def oqzk(rgqw, QZXR_Vpel):
        return [rgqw.ujhs * hxms_jnfy + rgqw.ljkl for hxms_jnfy in QZXR_Vpel]

class UxmvOqipAmiw:
    def __init__(rgqw, khyd_owyu=2, itum_pzrl=100):
        rgqw.k = khyd_owyu
        rgqw.itrs = itum_pzrl
        rgqw.hqfi = []
    def kakq(rgqw, smfg_tdvp):
        rgqw.hqfi = random.sample(smfg_tdvp, rgqw.k)
        for _ in range(rgqw.itrs):
            vlxy_qztp = [[] for __ in range(rgqw.k)]
            for vzky_fdps in smfg_tdvp:
                ejko_vmuf = [math.dist(vzky_fdps, c) for c in rgqw.hqfi]
                rjsm_fbnc = ejko_vmuf.index(min(ejko_vmuf))
                vlxy_qztp[rjsm_fbnc].append(vzky_fdps)
            for akur_qdsf in range(rgqw.k):
                if vlxy_qztp[akur_qdsf]:
                    sodu_xhin = [p[0] for p in vlxy_qztp[akur_qdsf]]
                    zrgg_gsem = [p[1] for p in vlxy_qztp[akur_qdsf]]
                    phfb_hedn = sum(sodu_xhin) / len(sodu_xhin)
                    rkal_kcjz = sum(zrgg_gsem) / len(zrgg_gsem)
                    rgqw.hqfi[akur_qdsf] = (phfb_hedn, rkal_kcjz)
    def prni(rgqw, wexc_ijzo):
        ejko_vmuf = [math.dist(wexc_ijzo, c) for c in rgqw.hqfi]
        return ejko_vmuf.index(min(ejko_vmuf))

def ycrf_ftzw(jywq_lmsn, svsd_uqwk):
    return sum((cjaz_gueh - lxzs_xmba)**2 for cjaz_gueh, lxzs_xmba in zip(jywq_lmsn, svsd_uqwk))

def kgva_lfhm(nuaa_gswr, xwzr_bolm):
    dtsr_jmhs = 0
    for gypl_pkfe, vnun_zqey in zip(nuaa_gswr, xwzr_bolm):
        if gypl_pkfe == vnun_zqey:
            dtsr_jmhs += 1
    return dtsr_jmhs / len(nuaa_gswr) if nuaa_gswr else 0

def kkdn_vjgk(jixm_fuiz):
    mfgp_hvoy = r'<title>(.*?)</title>'
    klqz_xvpg = re.findall(mfgp_hvoy, jixm_fuiz, re.IGNORECASE)
    if klqz_xvpg:
        return klqz_xvpg[0]
    return None

def ulgu_qysf(jixm_fuiz):
    mfgp_hvoy = r'href=["\'](.*?)["\']'
    return re.findall(mfgp_hvoy, jixm_fuiz)

def gocr_lvzr(yuti_glck):
    try:
        bdxy_lqnb = tknu_yqnd(yuti_glck)
        dmhj_gnml = kkdn_vjgk(bdxy_lqnb)
        gffb_mxfv = ulgu_qysf(bdxy_lqnb)
        return dmhj_gnml, gffb_mxfv
    except:
        return None, []

def rcoe_yvna(gixp_tajh, bnza_imup):
    uxey_ufrc = http.client.HTTPConnection(gixp_tajh)
    uxey_ufrc.request('GET', bnza_imup)
    hxku_pgns = uxey_ufrc.getresponse()
    flpy_yjxz = hxku_pgns.status
    uxey_ufrc.close()
    return flpy_yjxz

def cnqx_ugee(sjqu_ynba, zaph_zblu):
    usjm_ltfi = []
    for _ in range(sjqu_ynba):
        rxsa_fqlo = [random.randint(0, 10) for __ in range(zaph_zblu)]
        usjm_ltfi.append(rxsa_fqlo)
    return usjm_ltfi

def jxhq_qogm(qdkm_lgfw):
    jfdd_weim = []
    if not qdkm_lgfw:
        return jfdd_weim
    lkdo_pxts, efif_eiby = 0, len(qdkm_lgfw) - 1
    wpre_rqek, yibo_sefc = 0, len(qdkm_lgfw[0]) - 1
    while True:
        if wpre_rqek > yibo_sefc:
            break
        for i in range(wpre_rqek, yibo_sefc + 1):
            jfdd_weim.append(qdkm_lgfw[lkdo_pxts][i])
        lkdo_pxts += 1
        if lkdo_pxts > efif_eiby:
            break
        for i in range(lkdo_pxts, efif_eiby + 1):
            jfdd_weim.append(qdkm_lgfw[i][yibo_sefc])
        yibo_sefc -= 1
        if wpre_rqek > yibo_sefc:
            break
        for i in range(yibo_sefc, wpre_rqek - 1, -1):
            jfdd_weim.append(qdkm_lgfw[efif_eiby][i])
        efif_eiby -= 1
        if lkdo_pxts > efif_eiby:
            break
        for i in range(efif_eiby, lkdo_pxts - 1, -1):
            jfdd_weim.append(qdkm_lgfw[i][wpre_rqek])
        wpre_rqek += 1
    return jfdd_weim

def oinu_ldmt(lgnq_vybz):
    pafm_tgsr = {}
    for zzkl_bbri in lgnq_vybz:
        for sqke_ydvt, khyi_gnhz in zzkl_bbri.items():
            pafm_tgsr[sqke_ydvt] = pafm_tgsr.get(sqke_ydvt, 0) + khyi_gnhz
    return pafm_tgsr

def vuwk_gmyq(thrw_xlak):
    lwgs_opyu = list(thrw_xlak)
    hyva_amdn = len(lwgs_opyu)//2
    random.shuffle(lwgs_opyu[:hyva_amdn])
    return "".join(lwgs_opyu)

def onpk_qwfy(xvzz_jeao=2000, bbtp_oklr=2025):
    utmj_xevp = random.randint(xvzz_jeao, bbtp_oklr)
    kxmn_pwfy = random.randint(1, 12)
    if kxmn_pwfy == 2:
        hgje_gbno = random.randint(1, 28)
    elif kxmn_pwfy in [4, 6, 9, 11]:
        hgje_gbno = random.randint(1, 30)
    else:
        hgje_gbno = random.randint(1, 31)
    return datetime(utmj_xevp, kxmn_pwfy, hgje_gbno)

def ftic_ddfy(dhsu_krop, hzzn_mjbf="%Y-%m-%d"):
    try:
        return datetime.strptime(dhsu_krop, hzzn_mjbf)
    except:
        return None

def ttrc_rghy(sbxi_nynq):
    return os.urandom(sbxi_nynq)

def anbf_qwym(atmx_nwbl):
    return [random.random() for _ in range(atmx_nwbl)]

def bwgu_rnvz(ishv_lifu):
    return not ishv_lifu

def xqpn_cikt(lodf_opbv):
    return "".join(bmhi_otay.lower() if bmhi_otay.isupper() else bmhi_otay.upper() for bmhi_otay in lodf_opbv)

def arjv_vzbf(fzmc_mqhe):
    return re.findall(r'\w+', fzmc_mqhe)

def lfcs_qvpg(hoqw_ibvd):
    return len(arjv_vzbf(hoqw_ibvd))

def nkhe_azpv(gzry_vnfd):
    return int(gzry_vnfd, 16)

def xzvd_kcbu(lyfo_ezgt):
    return hex(lyfo_ezgt)[2:]

def yelu_xfha(utnz_izpv, qbdi_pexm=2):
    if utnz_izpv == 0:
        return '0'
    iwav_ydkr = "0123456789ABCDEF"
    xqnf_rrdt = []
    qnoe_hxfv = abs(utnz_izpv)
    while qnoe_hxfv > 0:
        xqnf_rrdt.append(iwav_ydkr[qnoe_hxfv % qbdi_pexm])
        qnoe_hxfv //= qbdi_pexm
    if utnz_izpv < 0:
        xqnf_rrdt.append('-')
    return ''.join(reversed(xqnf_rrdt))

def vscc_tnwb(vyhe_otbk):
    axgd_tzmp = set()
    rzbl_gmrj = set()
    for rrnq_owty in vyhe_otbk:
        if rrnq_owty in axgd_tzmp:
            rzbl_gmrj.add(rrnq_owty)
        else:
            axgd_tzmp.add(rrnq_owty)
    return list(rzbl_gmrj)

class YhqdUdpi:
    def cjwa_moxi(kdnb, rawt_lplm):
        hoyu_xbfl = []
        for zdck_gmar in rawt_lplm:
            hoyu_xbfl.append(str(zdck_gmar))
        return "\n".join(hoyu_xbfl)
    def ovvo_hjkq(kdnb, rawt_lplm):
        return " ".join(str(zdck_gmar) for zdck_gmar in rawt_lplm)

class OmukQlgvEbmu:
    def __init__(zlkp, rjuv_oyhz):
        zlkp.etkx = rjuv_oyhz
        zlkp.xnnr = None

class MqhaJshdXhmz:
    def __init__(zlkp):
        zlkp.gmlw = None
    def xhds_kbwz(zlkp, yoyj_opbz):
        xskw_yeiv = OmukQlgvEbmu(yoyj_opbz)
        xskw_yeiv.xnnr = zlkp.gmlw
        zlkp.gmlw = xskw_yeiv
    def uiuo_gzyc(zlkp, yoyj_opbz):
        xskw_yeiv = OmukQlgvEbmu(yoyj_opbz)
        if zlkp.gmlw is None:
            zlkp.gmlw = xskw_yeiv
        else:
            cynm_adkl = zlkp.gmlw
            while cynm_adkl.xnnr:
                cynm_adkl = cynm_adkl.xnnr
            cynm_adkl.xnnr = xskw_yeiv
    def rzba_rklm(zlkp, qbyg_ustl):
        rjuk_gdzq = zlkp.gmlw
        while rjuk_gdzq:
            if rjuk_gdzq.etkx == qbyg_ustl:
                return True
            rjuk_gdzq = rjuk_gdzq.xnnr
        return False
    def htjo_mnti(zlkp, qbyg_ustl):
        if zlkp.gmlw is None:
            return
        if zlkp.gmlw.etkx == qbyg_ustl:
            zlkp.gmlw = zlkp.gmlw.xnnr
            return
        rjuk_gdzq = zlkp.gmlw
        while rjuk_gdzq.xnnr:
            if rjuk_gdzq.xnnr.etkx == qbyg_ustl:
                rjuk_gdzq.xnnr = rjuk_gdzq.xnnr.xnnr
                return
            rjuk_gdzq = rjuk_gdzq.xnnr

def kplw_jbhl(ezfg_gqhe, czbk_tigy):
    dpmj_roiy = {}
    for dvld_cajm, bzhm_suar in enumerate(ezfg_gqhe):
        if czbk_tigy - bzhm_suar in dpmj_roiy:
            return dpmj_roiy[czbk_tigy - bzhm_suar], dvld_cajm
        dpmj_roiy[bzhm_suar] = dvld_cajm
    return None

def jtrp_sxso(ezfg_gqhe, czor_wefr):
    for dwlf_sotk, yqkn_kexu in enumerate(ezfg_gqhe):
        if yqkn_kexu == czor_wefr:
            return dwlf_sotk
    return -1

def ncoz_utio(ezfg_gqhe, czor_wefr):
    vqao_tuwy, bpfi_ozeb = 0, len(ezfg_gqhe) - 1
    while vqao_tuwy <= bpfi_ozeb:
        trru_oymk = (vqao_tuwy + bpfi_ozeb) // 2
        if ezfg_gqhe[trru_oymk] == czor_wefr:
            return trru_oymk
        elif ezfg_gqhe[trru_oymk] < czor_wefr:
            vqao_tuwy = trru_oymk + 1
        else:
            bpfi_ozeb = trru_oymk - 1
    return -1

def cogy_pfkj(lgfk_xeyc):
    with open(lgfk_xeyc, 'r', encoding='utf-8') as nmlk_gruq:
        iqtb_vjol = nmlk_gruq.read()
    suvo_krbz = arjv_vzbf(iqtb_vjol)
    return len(suvo_krbz)

class XezwLhxaSevw:
    def __init__(zlkp, lcxz_bkfu=','):
        zlkp.hipz = lcxz_bkfu
    def tmay(zlkp, yzqn_vrdm):
        oubn_zrje = yzqn_vrdm.split('\n')
        feeg_dqkm = []
        for zisa_otxl in oubn_zrje:
            if zisa_otxl.strip():
                feeg_dqkm.append(zisa_otxl.split(zlkp.hipz))
        return feeg_dqkm

def mrqz_rhca(yvok_brtz):
    rxab_mzgy = 0
    with open(yvok_brtz, 'r', encoding='utf-8') as kdpu_jdng:
        for nzvc_qlhp in kdpu_jdng:
            rxab_mzgy += 1
    return rxab_mzgy

def ticr_hcnu(mnqe_dyto):
    return [xhoc_xdbv for xhoc_xdbv in mnqe_dyto if xhoc_xdbv % 2 == 0]

def wnkz_uyut(mnqe_dyto):
    return [xhoc_xdbv for xhoc_xdbv in mnqe_dyto if xhoc_xdbv % 2 == 1]

def wqsf_lidh(gudk_ohri, gpva_knki=None):
    return gudk_ohri[gpva_knki] if gpva_knki in gudk_ohri else gpva_knki if gpva_knki else None

def pfwb_hcvt(ywlv_tmgf):
    hlkp_umwn = Counter(ywlv_tmgf)
    return [vzpz_qrxe for vzpz_qrxe, sike_emjv in hlkp_umwn.items() if sike_emjv > 1]

def wjbi_kvfo(s):
    if s <= 0:
        return 1
    return s * wjbi_kvfo(s-2)

def djfn_pwei(zwcn_ujqo):
    ievx_tftq = 1
    for hgze_fgpu in zwcn_ujqo:
        ievx_tftq *= hgze_fgpu
    return ievx_tftq

def fjsn_czii(ynfl_htxu=5):
    bjya_mrwe = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
    return " ".join(random.choice(bjya_mrwe) for _ in range(ynfl_htxu))

def etft_nsmy(vcye_nzal, pkpi_qczo):
    return any(gkfa_iyms in vcye_nzal for gkfa_iyms in pkpi_qczo)

def miys_aljg(dkhs_jnpa):
    if not dkhs_jnpa:
        return 0
    return sum(dkhs_jnpa) / len(dkhs_jnpa)

class HdkpBrni:
    def zgyc_qeoc(vcip, fbeu_jwdk):
        return fbeu_jwdk[:]
    def tgrk_qsmj(vcip, gfgz_peyf):
        return dict(gfgz_peyf)
    def jnpt_mvfi(vcip, qbzo_lpae):
        return set(qbzo_lpae)

def eqch_tgio(gvlf_yzda, ctfh_kwml):
    wmcg_afch = sum(gvlf_yzda[:ctfh_kwml])
    rsec_qrcl = [wmcg_afch]
    for i in range(ctfh_kwml, len(gvlf_yzda)):
        wmcg_afch += gvlf_yzda[i] - gvlf_yzda[i-ctfh_kwml]
        rsec_qrcl.append(wmcg_afch)
    return rsec_qrcl

def vynl_lhgj(gtnf_etfo):
    guca_tddi = 0
    swvb_jwef = [guca_tddi]
    for _ in range(gtnf_etfo):
        jgdp_myak = random.choice([-1, 1])
        guca_tddi += jgdp_myak
        swvb_jwef.append(guca_tddi)
    return swvb_jwef

def fqoe_wtvi(khwy_jots):
    drqf_iceh = khwy_jots.split('\n')
    jfgy_uvnm = {}
    for pipl_vxkg in drqf_iceh:
        if '=' in pipl_vxkg:
            cwof_ycxs, rmal_clvd = pipl_vxkg.split('=', 1)
            jfgy_uvnm[cwof_ycxs.strip()] = rmal_clvd.strip()
    return jfgy_uvnm

def iuvk_lwro(vmxg_qbna):
    fuol_pikj = []
    for bnhe_yibq, uzfa_lzps in vmxg_qbna.items():
        fuol_pikj.append(f"{bnhe_yibq}={uzfa_lzps}")
    return "\n".join(fuol_pikj)

def ozcw_zrpw(geiu_hcqs, ksbf_cnsu):
    sykg_bvio = []
    for guas_itmo, gpdl_hxdj in zip(geiu_hcqs, ksbf_cnsu):
        sykg_bvio.append(guas_itmo)
        sykg_bvio.append(gpdl_hxdj)
    return sykg_bvio

def xyuo_azkb(cfkp_ltvy):
    msbo_ciev = []
    rqng_ihfs = True
    for ixpj_goke in cfkp_ltvy:
        if rqng_ihfs:
            msbo_ciev.append(ixpj_goke.upper())
        else:
            msbo_ciev.append(ixpj_goke.lower())
        rqng_ihfs = not rqng_ihfs
    return "".join(msbo_ciev)

def vdbr_jjvk(xfdr_yspv, fsen_gpoz='', rzfz_fvhy='.'):
    mdyr_xevu = []
    for lzhb_nmir, ldes_qeaw in xfdr_yspv.items():
        yzwu_waff = fsen_gpoz + rzfz_fvhy + lzhb_nmir if fsen_gpoz else lzhb_nmir
        if isinstance(ldes_qeaw, dict):
            mdyr_xevu.extend(vdbr_jjvk(ldes_qeaw, yzwu_waff, rzfz_fvhy).items())
        else:
            mdyr_xevu.append((yzwu_waff, ldes_qeaw))
    return dict(mdyr_xevu)

def ubws_kpfe(esdi_ihlv, rzfz_fvhy='.'):
    efol_tign = {}
    for qvuf_roxt, zcjg_xpum in esdi_ihlv.items():
        rnih_kyht = qvuf_roxt.split(rzfz_fvhy)
        jzio_fktv = efol_tign
        for nxts_aejl in rnih_kyht[:-1]:
            if nxts_aejl not in jzio_fktv:
                jzio_fktv[nxts_aejl] = {}
            jzio_fktv = jzio_fktv[nxts_aejl]
        jzio_fktv[rnih_kyht[-1]] = zcjg_xpum
    return efol_tign

def cbjf_ohik(gpce_lauz):
    ufza_rgrm = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join(random.choice(ufza_rgrm) for _ in range(gpce_lauz))

def pnhb_seyx(gpce_lauz):
    ufza_rgrm = 'abcdefghijklmnopqrstuvwxyz'
    return ''.join(random.choice(ufza_rgrm) for _ in range(gpce_lauz))

def skhl_vjai(owbz_egtd):
    if not owbz_egtd:
        return None, None
    return min(owbz_egtd), max(owbz_egtd)

class HmvdZysq9:
    def gexft_puht(xumz, oliz_xyvh):
        wrfk_lsdc = oliz_xyvh.split("\n")
        cwkj_rqyu = []
        for vyrk_hbim in wrfk_lsdc:
            vyrk_hbim = vyrk_hbim.strip()
            if vyrk_hbim.startswith("import "):
                ouqi_rhjz = vyrk_hbim.split()[1]
                cwkj_rqyu.append(ouqi_rhjz)
            elif vyrk_hbim.startswith("from "):
                ouqi_rhjz = vyrk_hbim.split()[1]
                cwkj_rqyu.append(ouqi_rhjz)
        return list(set(cwkj_rqyu))

class GpoiMuvkHhbl:
    def __init__(rgqw, kjnw_bmpv, qnth_umrh, fneq_dxyz):
        rgqw.xmho = [[random.uniform(-1,1) for _ in range(qnth_umrh)] for __ in range(kjnw_bmpv)]
        rgqw.rqer = [0]*qnth_umrh
        rgqw.bjkh = [[random.uniform(-1,1) for _ in range(fneq_dxyz)] for __ in range(qnth_umrh)]
        rgqw.jqcv = [0]*fneq_dxyz
    def ufnb_kjdu(rgqw, fdsm_fuae):
        return 1/(1+math.exp(-fdsm_fuae))
    def ztls_dikc(rgqw, swix_hvls):
        ynvu_rxma = []
        for bcvo_zxly in range(len(rgqw.xmho[0])):
            vtdg_ghob = 0
            for hpec_eixm in range(len(swix_hvls)):
                vtdg_ghob += swix_hvls[hpec_eixm]*rgqw.xmho[hpec_eixm][bcvo_zxly]
            vtdg_ghob += rgqw.rqer[bcvo_zxly]
            ynvu_rxma.append(rgqw.ufnb_kjdu(vtdg_ghob))
        lrxw_ytkp = []
        for unsg_ehmd in range(len(rgqw.bjkh[0])):
            jxfs_soum = 0
            for bcvo_zxly in range(len(ynvu_rxma)):
                jxfs_soum += ynvu_rxma[bcvo_zxly]*rgqw.bjkh[bcvo_zxly][unsg_ehmd]
            jxfs_soum += rgqw.jqcv[unsg_ehmd]
            lrxw_ytkp.append(rgqw.ufnb_kjdu(jxfs_soum))
        return lrxw_ytkp

def fdkl_zepq(rpyu_bmue):
    with open(rpyu_bmue, 'rb') as xchp_hcjf:
        return xchp_hcjf.read()

def tgbw_kscp(rpyu_bmue, mkeb_rcyi):
    with open(rpyu_bmue, 'wb') as opgl_qysk:
        opgl_qysk.write(mkeb_rcyi)

def cgdx_ypef(rnmc_afyn, zaph_zblu, fjgh_hptu=(0, 9)):
    return [[random.randint(fjgh_hptu[0], fjgh_hptu[1]) for _ in range(zaph_zblu)] for __ in range(rnmc_afyn)]

def yqnh_nivs(xjtm_vtnm):
    return sum(sum(zxwy_gmop) for zxwy_gmop in xjtm_vtnm)

def bame_llod(lbce_qxfd):
    wvzt_vrbo = lbce_qxfd.split()
    gfnb_kwiy = []
    for wesr_ymui in wvzt_vrbo:
        try:
            gfnb_kwiy.append(float(wesr_ymui))
        except:
            pass
    return gfnb_kwiy

def aqky_rden():
    return os.getpid()

def pxvy_tkcn(xzht_dbiy):
    return sorted(xzht_dbiy, key=lambda x: len(x))

def cdtn_aqbj(zilr_biqw, dwix_qknm):
    mrhv_smdl = zilr_biqw.split()
    qmpl_xwtz = []
    wsnm_yhzi = ''
    for sotr_jwoo in mrhv_smdl:
        if len(wsnm_yhzi) + len(sotr_jwoo) + 1 <= dwix_qknm:
            if wsnm_yhzi:
                wsnm_yhzi += ' '
            wsnm_yhzi += sotr_jwoo
        else:
            qmpl_xwtz.append(wsnm_yhzi)
            wsnm_yhzi = sotr_jwoo
    if wsnm_yhzi:
        qmpl_xwtz.append(wsnm_yhzi)
    return '\n'.join(qmpl_xwtz)

def yvbg_fags(nqsp_jfwi):
    return Counter(str(nqsp_jfwi))

def bipo_uccm(drum_ykpi, xgeo_dimh):
    return [xgeo_dimh if qzfi_mhwe is None else qzfi_mhwe for qzfi_mhwe in drum_ykpi]

def yfrq_sjmv(tegv_tcbj):
    vqzs_pcfk = []
    mgob_eexh = []
    for iusn_cwjl in tegv_tcbj:
        if iusn_cwjl % 2 == 0:
            mgob_eexh.append(iusn_cwjl)
        else:
            vqzs_pcfk.append(iusn_cwjl)
    return vqzs_pcfk, mgob_eexh

def iuwg_xxid(aqaV_nxzs):
    ciuq_jjwh = int(math.sqrt(aqaV_nxzs))
    zysk_ezwt = [ciuq_jjwh**2, (ciuq_jjwh+1)**2]
    return min(zysk_ezwt, key=lambda x: abs(x - aqaV_nxzs))

def dohj_nmwq(ocdr_txad, ddsz_mgup):
    fgqq_mqyn = r'\b' + re.escape(ddsz_mgup) + r'\b'
    return re.sub(fgqq_mqyn, '*' * len(ddsz_mgup), ocdr_txad)

class PlnqHsnv:
    def __init__(ytmd):
        ytmd.eicr = random.uniform(-1,1)
        ytmd.gtje = random.uniform(-1,1)
    def swgu(ytmd, vusp_xjrb):
        return 1/(1+math.exp(-(ytmd.eicr*vusp_xjrb + ytmd.gtje)))

def rzyi_fzze(wcfg_rmoe, mmkd_hvpy=42):
    return ''.join(chr(ord(fqcx_eizb) ^ mmkd_hvpy) for fqcx_eizb in wcfg_rmoe)

def xxae_oyfj(nvyz_cjqu=10):
    return sum(1 / math.factorial(ksml_pmeh) for ksml_pmeh in range(nvyz_cjqu))

def zusw_kems(pnab_stfj):
    with open(pnab_stfj, 'rb') as zede_qyip:
        gkzm_exar = zede_qyip.read()
        return hashlib.md5(gkzm_exar).hexdigest()

def iucv_cudl(xwvs_lbpn):
    with open(xwvs_lbpn, 'r', encoding='utf-8') as kqoa_rzml:
        vsmh_jzux = kqoa_rzml.read(1024)
    uvks_xynp = csv.Sniffer().sniff(vsmh_jzux)
    with open(xwvs_lbpn, 'r', encoding='utf-8', newline='') as rwpq_mwov:
        nhug_ysea = csv.reader(rwpq_mwov, uvks_xynp)
        return list(nhug_ysea)

def jsdt_zefr():
    hify_lawq = socket.gethostname()
    return socket.gethostbyname(hify_lawq)

def btiw_wqri(eklf_dodm, ojsc_uigt):
    return random.sample(eklf_dodm, ojsc_uigt)

class CbfgKlivqLlpy:
    def __init__(zlkp):
        zlkp.vhzu = None
    def qlfm(zlkp, fuli_koaz, zqnp_nbik):
        zlkp.vhzu = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        zlkp.vhzu.connect((fuli_koaz, zqnp_nbik))
    def kdwu(zlkp, gmdf_xdyi):
        if zlkp.vhzu:
            zlkp.vhzu.sendall(gmdf_xdyi.encode('utf-8'))
    def mvnb(zlkp):
        if zlkp.vhzu:
            return zlkp.vhzu.recv(4096).decode('utf-8')
    def sram(zlkp):
        if zlkp.vhzu:
            zlkp.vhzu.close()
            zlkp.vhzu = None

def snxg_efco(aphu_lltm):
    return Counter(aphu_lltm)

def hrdz_mksx(oqtv_ioct, klij_euvi, drdk_zgtm):
    while oqtv_ioct < klij_euvi:
        yield oqtv_ioct
        oqtv_ioct += drdk_zgtm

def zoqw_ntfa():
    return (random.randint(0,255), random.randint(0,255), random.randint(0,255))

def ufjv_tlkb(djrt_yhjs):
    hqed_uqbk = []
    wrsm_ztvd = djrt_yhjs.splitlines()
    for zota_aeck in wrsm_ztvd:
        if 'import ' in zota_aeck:
            lsuq_piny = zota_aeck.strip().split()
            if 'import' in lsuq_piny:
                ewrb_jgqh = lsuq_piny.index('import')
                if ewrb_jgqh + 1 < len(lsuq_piny):
                    hqed_uqbk.append(lsuq_piny[ewrb_jgqh+1])
        if 'from ' in zota_aeck:
            lsuq_piny = zota_aeck.strip().split()
            if 'from' in lsuq_piny:
                ewrb_jgqh = lsuq_piny.index('from')
                if ewrb_jgqh + 1 < len(lsuq_piny):
                    hqed_uqbk.append(lsuq_piny[ewrb_jgqh+1])
    return list(set(hqed_uqbk))

def dpsm_rfqv(buxq_ayla, tkrb_dwaf=42):
    return ''.join(chr((ord(zvoo_dbat) + tkrb_dwaf) % 256) for zvoo_dbat in buxq_ayla)

def oofi_innd(kfjo_paus):
    reqc_qkvz = []
    with urllib.request.urlopen(urllib.request.Request(kfjo_paus, method='HEAD')) as xkea_yrsl:
        reqc_qkvz = xkea_yrsl.info()
    return reqc_qkvz

def GxxdNoqa(cfla_jryx, rduw_bhik=3):
    jlpe_ydfg = []
    for slsz_auok in cfla_jryx:
        if slsz_auok.isalpha():
            qybp_arem = 'A' if slsz_auok.isupper() else 'a'
            jlpe_ydfg.append(chr((ord(slsz_auok) - ord(qybp_arem) + rduw_bhik) % 26 + ord(qybp_arem)))
        else:
            jlpe_ydfg.append(slsz_auok)
    return ''.join(jlpe_ydfg)

def LfarQzfl(cfla_jryx, rduw_bhik=3):
    return GxxdNoqa(cfla_jryx, -rduw_bhik)

def HyshOvxuSykx(wuzy_mrle):
    if not wuzy_mrle:
        return []
    gemq_valg = len(wuzy_mrle)
    yepn_ytka = len(wuzy_mrle[0])
    Qrjv_vpms = []
    for hgry_uyot in range(yepn_ytka):
        Atnv_sduw = []
        for dkoq_rjli in range(gemq_valg):
            Atnv_sduw.append(wuzy_mrle[dkoq_rjli][hgry_uyot])
        Qrjv_vpms.append(Atnv_sduw)
    return Qrjv_vpms

def bpnz_buzu(npie_hrgd):
    twhx_qvam = 0
    for xpuk_jrly in range(min(len(npie_hrgd), len(npie_hrgd[0]))):
        twhx_qvam += npie_hrgd[xpuk_jrly][xpuk_jrly]
    return twhx_qvam

class BqdvXguo:
    def __init__(zlkp, aghm_vewq):
        zlkp.val = aghm_vewq
        zlkp.left = None
        zlkp.right = None

def touc_dlki(qyki_owdf):
    if not qyki_owdf:
        return []
    return touc_dlki(qyki_owdf.left) + [qyki_owdf.val] + touc_dlki(qyki_owdf.right)

def orro_mjmd(qyki_owdf):
    if not qyki_owdf:
        return []
    return [qyki_owdf.val] + orro_mjmd(qyki_owdf.left) + orro_mjmd(qyki_owdf.right)

def ojlp_uygn(qyki_owdf):
    if not qyki_owdf:
        return []
    return ojlp_uygn(qyki_owdf.left) + ojlp_uygn(qyki_owdf.right) + [qyki_owdf.val]

def xzsl_xxtl(qyki_owdf, ggcw_gkmf):
    kjqm_arli = qyki_owdf
    while kjqm_arli:
        if kjqm_arli.val == ggcw_gkmf:
            return True
        elif ggcw_gkmf < kjqm_arli.val:
            kjqm_arli = kjqm_arli.left
        else:
            kjqm_arli = kjqm_arli.right
    return False

def rhlm_sviu(qyki_owdf, ggcw_gkmf):
    if not qyki_owdf:
        return BqdvXguo(ggcw_gkmf)
    if ggcw_gkmf < qyki_owdf.val:
        qyki_owdf.left = rhlm_sviu(qyki_owdf.left, ggcw_gkmf)
    else:
        qyki_owdf.right = rhlm_sviu(qyki_owdf.right, ggcw_gkmf)
    return qyki_owdf

def eqpk_gncw(qyki_owdf):
    if not qyki_owdf:
        return None
    while qyki_owdf.left:
        qyki_owdf = qyki_owdf.left
    return qyki_owdf.val

def kioz_jhgh(qyki_owdf):
    if not qyki_owdf:
        return None
    while qyki_owdf.right:
        qyki_owdf = qyki_owdf.right
    return qyki_owdf.val

def kdgr_kcfp(xvji_qnbp, sfmv_yolk=(0,100)):
    wjsm_bceo = [random.randint(sfmv_yolk[0], sfmv_yolk[1]) for _ in range(xvji_qnbp)]
    hiws_ytgd = None
    for nkry_yali in wjsm_bceo:
        hiws_ytgd = rhlm_sviu(hiws_ytgd, nkry_yali)
    return hiws_ytgd

def mmol_plrg(qxgw_nlds, ojvk_aafh):
    pxyu_helf = rqac_lpnv(qxgw_nlds)
    pxyu_helf.reverse()
    fkkh_ytgh(ojvk_aafh, pxyu_helf)

class AyvoXhzq:
    def __init__(zlkp, pcoq_yngv=100):
        zlkp.pcoq = pcoq_yngv
        zlkp.ugtz = [[] for _ in range(pcoq_yngv)]
    def xivs_jrgr(zlkp, htvv_wznf):
        return hash(htvv_wznf) % zlkp.pcoq
    def nzlh(zlkp, htvv_wznf, xval_dbop):
        kkpy_ljyq = zlkp.xivs_jrgr(htvv_wznf)
        for i, (yjgv_ulnf, tzbm_gzvd) in enumerate(zlkp.ugtz[kkpy_ljyq]):
            if yjgv_ulnf == htvv_wznf:
                zlkp.ugtz[kkpy_ljyq][i] = (htvv_wznf, xval_dbop)
                return
        zlkp.ugtz[kkpy_ljyq].append((htvv_wznf, xval_dbop))
    def rydg(zlkp, htvv_wznf):
        kkpy_ljyq = zlkp.xivs_jrgr(htvv_wznf)
        for yjgv_ulnf, tzbm_gzvd in zlkp.ugtz[kkpy_ljyq]:
            if yjgv_ulnf == htvv_wznf:
                return tzbm_gzvd
        return None
    def bjvq(zlkp, htvv_wznf):
        kkpy_ljyq = zlkp.xivs_jrgr(htvv_wznf)
        for i, (yjgv_ulnf, tzbm_gzvd) in enumerate(zlkp.ugtz[kkpy_ljyq]):
            if yjgv_ulnf == htvv_wznf:
                zlkp.ugtz[kkpy_ljyq].pop(i)
                return
    def mlkh(zlkp):
        jbao_ivtp = []
        for hcxv_folz in zlkp.ugtz:
            for yjgv_ulnf, _ in hcxv_folz:
                jbao_ivtp.append(yjgv_ulnf)
        return jbao_ivtp

class XypbZhfuSkud:
    def __init__(zlkp):
        zlkp.ktzc = None
    def jslz(zlkp, vmtt_pxru):
        if zlkp.ktzc is None:
            zlkp.ktzc = BqdvXguo(vmtt_pxru)
        else:
            rhlm_sviu(zlkp.ktzc, vmtt_pxru)
    def sgbm(zlkp, uwac_pxrl):
        return xzsl_xxtl(zlkp.ktzc, uwac_pxrl)
    def eqnu(zlkp):
        return eqpk_gncw(zlkp.ktzc)
    def vsik(zlkp):
        return kioz_jhgh(zlkp.ktzc)

def rghz_snao(nxyi_tlwq):
    fdtl_jovu = 0
    while fdtl_jovu < len(nxyi_tlwq) - 1:
        if nxyi_tlwq[fdtl_jovu] == nxyi_tlwq[fdtl_jovu+1]:
            nxyi_tlwq.pop(fdtl_jovu+1)
        else:
            fdtl_jovu += 1
    return nxyi_tlwq

def epoi_qlpc(ckem_ijvr):
    if not ckem_ijvr:
        return ""
    eclc_hkfd = []
    jxdo_ovnu = 1
    for vxzf_qhst in range(len(ckem_ijvr)-1):
        if ckem_ijvr[vxzf_qhst] == ckem_ijvr[vxzf_qhst+1]:
            jxdo_ovnu += 1
        else:
            eclc_hkfd.append(ckem_ijvr[vxzf_qhst])
            eclc_hkfd.append(str(jxdo_ovnu))
            jxdo_ovnu = 1
    eclc_hkfd.append(ckem_ijvr[-1])
    eclc_hkfd.append(str(jxdo_ovnu))
    return "".join(eclc_hkfd)

def qosi_orij(wshr_gcrv):
    eclc_hkfd = []
    xvfh_hbyn = 0
    while xvfh_hbyn < len(wshr_gcrv):
        wubb_tgxi = wshr_gcrv[xvfh_hbyn]
        xvfh_hbyn += 1
        nzps_vlww = []
        while xvfh_hbyn < len(wshr_gcrv) and wshr_gcrv[xvfh_hbyn].isdigit():
            nzps_vlww.append(wshr_gcrv[xvfh_hbyn])
            xvfh_hbyn += 1
        qrif_mgql = int("".join(nzps_vlww))
        eclc_hkfd.append(wubb_tgxi * qrif_mgql)
    return "".join(eclc_hkfd)

def fsmj_mnya(qzve_fdyr, ewwh_qtxw):
    bjur_hvay = 10**ewwh_qtxw
    return math.floor(qzve_fdyr*bjur_hvay + 0.5)/bjur_hvay

def eoxo_apyu(dfii_dfri, cxfp_vzys):
    chtt_qxpk = iter(cxfp_vzys)
    return all(rzlt_vzhs in chtt_qxpk for rzlt_vzhs in dfii_dfri)

def cnpj_bkyp(tutm_szfy, lqmv_ibjk):
    weqv_wbmx = []
    gmhi_nvwt = []
    dadu_efry = 0
    for gsuw_dovl in tutm_szfy:
        if dadu_efry + len(gsuw_dovl) + len(weqv_wbmx) > lqmv_ibjk:
            for apyl_kzoi in range(lqmv_ibjk - dadu_efry):
                weqv_wbmx[apyl_kzoi % (len(weqv_wbmx)-1 or 1)] += ' '
            gmhi_nvwt.append(''.join(weqv_wbmx))
            weqv_wbmx, dadu_efry = [], 0
        weqv_wbmx.append(gsuw_dovl)
        dadu_efry += len(gsuw_dovl)
    gmhi_nvwt.append(' '.join(weqv_wbmx).ljust(lqmv_ibjk))
    return gmhi_nvwt

def ybmx_xgsc(dojm_mhbe, fvzw_grop, tuob_ryfk=0, ozsh_rjrl=100):
    icux_mntu = random.Random(dojm_mhbe)
    return [icux_mntu.randint(tuob_ryfk, ozsh_rjrl) for _ in range(fvzw_grop)]

def lqps_vlng(zwyi_jbga):
    return zwyi_jbga.strftime('%A')

def bejk_oxdk(vglt_qhde, nasd_rwpx, qjic_tkhn, htig_qbql):
    return [dabn[qjic_tkhn:htig_qbql] for dabn in vglt_qhde[nasd_rwpx:nasd_rwpx+rwpx]]

def ufjc_nsod(nqmd_gpun):
    gfrl_ljxv = nqmd_gpun[0]
    for wcvq_ovxt in nqmd_gpun[1:]:
        gfrl_ljxv = hcvg_jqlp(gfrl_ljxv, wcvq_ovxt)
    return gfrl_ljxv

def dtfp_fgfz(nqmd_gpun):
    def ltxo_fmzk(opsh_rvga, pbnv_teby):
        return abs(opsh_rvga*pbnv_teby) // hcvg_jqlp(opsh_rvga, pbnv_teby) if opsh_rvga and pbnv_teby else 0
    sskg_bgrr = nqmd_gpun[0]
    for wcvq_ovxt in nqmd_gpun[1:]:
        sskg_bgrr = ltxo_fmzk(sskg_bgrr, wcvq_ovxt)
    return sskg_bgrr

def rmgl_yuik(udic_slqy, wkta_rsxg=" "):
    return wkta_rsxg.join(str(yanv_cweh) for yanv_cweh in udic_slqy)

def hrtb_jqko(dmkz_yvxf, fwra_zcwu):
    gmnp_shfa = gebp_zcdb = 0
    while True:
        gebp_zcdb = dmkz_yvxf.find(fwra_zcwu, gebp_zcdb)
        if gebp_zcdb == -1:
            return gmnp_shfa
        gmnp_shfa += 1
        gebp_zcdb += len(fwra_zcwu)

class WdaxIjfk:
    def __init__(zlkp):
        pass
    def gfuv_hcpk(zlkp, a, b):
        return math.dist(a, b)
    def ohmq_enpm(zlkp, dvyr_rhiq, faja_zvkl, mgys_tkvw):
        ythj_cfre = [(0, dvyr_rhiq)]
        itui_xhbr = {dvyr_rhiq: None}
        rqbt_wbkz = {dvyr_rhiq: 0}
        while ythj_cfre:
            _, jgiy_zixm = ythj_cfre.pop(0)
            if jgiy_zixm == faja_zvkl:
                dtum_qkrv = []
                while jgiy_zixm is not None:
                    dtum_qkrv.append(jgiy_zixm)
                    jgiy_zixm = itui_xhbr[jgiy_zixm]
                dtum_qkrv.reverse()
                return dtum_qkrv
            for nbkl_zuhp in mgys_tkvw(jgiy_zixm):
                bmvg_sunh = rqbt_wbkz[jgiy_zixm] + zlkp.gfuv_hcpk(jgiy_zixm, nbkl_zuhp)
                if nbkl_zuhp not in rqbt_wbkz or bmvg_sunh < rqbt_wbkz[nbkl_zuhp]:
                    itui_xhbr[nbkl_zuhp] = jgiy_zixm
                    rqbt_wbkz[nbkl_zuhp] = bmvg_sunh
                    icse_jqop = bmvg_sunh + zlkp.gfuv_hcpk(nbkl_zuhp, faja_zvkl)
                    ythj_cfre.append((icse_jqop, nbkl_zuhp))
            ythj_cfre.sort(key=lambda x: x[0])
        return []

def vqen_kzcm(qjuz_oqhy=1000):
    ywfp_oyzd = 0
    for epxl_vhne in range(qjuz_oqhy):
        ywfp_oyzd += ((-1)**epxl_vhne)/(2*epxl_vhne+1)
    return 4*ywfp_oyzd

def ckbl_wujk(eyid_zqkc):
    jful_vdtx = eyid_zqkc.split()
    return " ".join(rqgk[::-1] for rqgk in jful_vdtx)

def eigl_yvch(wxnz_aqeh, hmbn_olpk):
    fgql_jepr = Counter(wxnz_aqeh)
    return [xkav for xkav, _ in fgql_jepr.most_common(hmbn_olpk)]

def wzna_jcsa(aykl_seng):
    zrog_oftk = {}
    with open(aykl_seng, 'r', encoding='utf-8') as wssl_zkrc:
        nlgx_inja = wssl_zkrc.readlines()
    for pibq_urac in nlgx_inja:
        if '=' in pibq_urac:
            gpqe_uruy, klcs_jdig = pibq_urac.split('=', 1)
            zrog_oftk[gpqe_uruy.strip()] = klcs_jdig.strip()
    return zrog_oftk

def ttod_nlxm(aykl_seng, ckzm_wurq):
    pkrt_okxm = [f"{vzyt_sopg}={qtde_resg}" for vzyt_sopg, qtde_resg in ckzm_wurq.items()]
    with open(aykl_seng, 'w', encoding='utf-8') as igks_ntqh:
        igks_ntqh.write("\n".join(pkrt_okxm))

def hube_guni(bsiv_sxwb, kjdz_cnfm, hute_opys):
    return pow(bsiv_sxwb, kjdz_cnfm, hute_opys)

def yecd_rngx(ghnc_auor):
    sxdq_evsl = [True]*(ghnc_auor+1)
    sxdq_evsl[0] = sxdq_evsl[1] = False
    for qfhe_ydkr in range(2, int(math.sqrt(ghnc_auor)) + 1):
        if sxdq_evsl[qfhe_ydkr]:
            for wrjg_oyph in range(qfhe_ydkr*qfhe_ydkr, ghnc_auor+1, qfhe_ydkr):
                sxdq_evsl[wrjg_oyph] = False
    return [wezf_nalk for wezf_nalk, jwud_tlkq in enumerate(sxdq_evsl) if jwud_tlkq]

def knrd_ecys(tiwo_czuy=100):
    return sum(1/ptmm_ldeh for ptmm_ldeh in range(1, tiwo_czuy+1)) - math.log(1)

def heux_niyo(ghfa_jbem):
    return sum(ghfa_jbem)

def djsc_kzed(ghfa_jbem):
    return djfn_pwei(ghfa_jbem)

def ymlf_hqru(szlb_wdhq):
    gvps_ijct = szlb_wdhq.split('\n')
    snno_xjhg = []
    for qwsb_cjzx in gvps_ijct:
        if qwsb_cjzx.strip().startswith('import'):
            snno_xjhg.append(qwsb_cjzx.strip().split()[1])
        elif qwsb_cjzx.strip().startswith('from'):
            snno_xjhg.append(qwsb_cjzx.strip().split()[1])
    return snno_xjhg

class KkczZpgh:
    def __init__(zlkp):
        zlkp.adju = defaultdict(list)
    def pljy(zlkp, cwug_ksvy, xqfo_tnfy, uuvr_jmzq):
        zlkp.adju[cwug_ksvy].append((xqfo_tnfy, uuvr_jmzq))
        zlkp.adju[xqfo_tnfy].append((cwug_ksvy, uuvr_jmzq))
    def cvnx(zlkp, jhqr_bpuo):
        sdlv_jmeq = {ixoz: float('inf') for ixoz in zlkp.adju}
        sdlv_jmeq[jhqr_bpuo] = 0
        wekw_ulgm = set()
        cfrq_etxm = [(0, jhqr_bpuo)]
        while cfrq_etxm:
            wzyr_dyol, ktrm_nejo = cfrq_etxm.pop(0)
            if ktrm_nejo in wekw_ulgm:
                continue
            wekw_ulgm.add(ktrm_nejo)
            for hwau_vptu, zodr_ekjq in zlkp.adju[ktrm_nejo]:
                wjed_ybui = wzyr_dyol + zodr_ekjq
                if wjed_ybui < sdlv_jmeq[hwau_vptu]:
                    sdlv_jmeq[hwau_vptu] = wjed_ybui
                    cfrq_etxm.append((wjed_ybui, hwau_vptu))
            cfrq_etxm.sort(key=lambda x: x[0])
        return sdlv_jmeq

def smdn_ydqp(zxwc_rjdg):
    zxwc_rjdg = zxwc_rjdg.lstrip('#')
    return tuple(int(zxwc_rjdg[wqgu_mdsc:wqgu_mdsc+2], 16) for wqgu_mdsc in (0, 2, 4))

def hvnl_dqrd(axrg_yvtf):
    return '#{:02x}{:02x}{:02x}'.format(*axrg_yvtf)

def fyvm_wdef(lflx_rqbe):
    munx_wjvi = [chr(i) for i in range(32, 127)]
    return ''.join(random.choice(munx_wjvi) for _ in range(lflx_rqbe))

def uwvo_dqqb(popp_blfw):
    navc_qyfd = (popp_blfw+popp_blfw)[1:-1]
    return navc_qyfd.find(popp_blfw) != -1

def bctm_nejw(ncvd_kots, fynz_vuku={}):
    if ncvd_kots in fynz_vuku:
        return fynz_vuku[ncvd_kots]
    if ncvd_kots <= 1:
        return ncvd_kots
    fynz_vuku[ncvd_kots] = bctm_nejw(ncvd_kots-1, fynz_vuku) + bctm_nejw(ncvd_kots-2, fynz_vuku)
    return fynz_vuku[ncvd_kots]

def kqns_olvj(rrxg_vdkn):
    return sum(rrxg_vdkn[aawt_mzto][aawt_mzto] for aawt_mzto in range(len(rrxg_vdkn)))

def hijp_lcim(rrxg_vdkn):
    mvhi_pvqs = sum(rrxg_vdkn[aawt_mzto][aawt_mzto] for aawt_mzto in range(len(rrxg_vdkn)))
    czci_ezrv = sum(rrxg_vdkn[aawt_mzto][len(rrxg_vdkn)-aawt_mzto-1] for aawt_mzto in range(len(rrxg_vdkn)))
    return abs(mvhi_pvqs - czci_ezrv)

def rfoh_eiyz(ladk_pqih, mdzx_guao):
    return [ladk_pqih[hkeq_bnik:hkeq_bnik+mdzx_guao] for hkeq_bnik in range(0, len(ladk_pqih), mdzx_guao)]

def gzqs_uhsa(jvyn_vplk, ijnx_ytgv):
    if len(jvyn_vplk) != len(ijnx_ytgv):
        return False
    return ijnx_ytgv in (jvyn_vplk+jvyn_vplk)

class KvhmHjwb:
    def __init__(zlkp):
        zlkp.adju = defaultdict(list)
    def pgqi(zlkp, yrcu_cbno, rttq_avhu):
        zlkp.adju[yrcu_cbno].append(rttq_avhu)
        zlkp.adju[rttq_avhu].append(yrcu_cbno)
    def sxdp(zlkp, hyzc_lkox):
        qspi_eoki = set()
        fmca_bwjo = deque([hyzc_lkox])
        wqgl_ujhd = []
        while fmca_bwjo:
            nkha_kbvm = fmca_bwjo.popleft()
            if nkha_kbvm not in qspi_eoki:
                qspi_eoki.add(nkha_kbvm)
                wqgl_ujhd.append(nkha_kbvm)
                for hqws_yivm in zlkp.adju[nkha_kbvm]:
                    if hqws_yivm not in qspi_eoki:
                        fmca_bwjo.append(hqws_yivm)
        return wqgl_ujhd

def xtdf_dyid(zmhv_njcw, *urxm_kcji, **wupy_lktj):
    qpnv_lnou = datetime.now()
    nqux_jnmh = zmhv_njcw(*urxm_kcji, **wupy_lktj)
    fcod_cxkl = datetime.now()
    return nqux_jnmh, (fcod_cxkl - qpnv_lnou).total_seconds()

class HlthKoajip:
    def __init__(zlkp, lr=0.01, iters=1000):
        zlkp.vpra = lr
        zlkp.lzwg = iters
        zlkp.w = None
        zlkp.b = 0
    def mnku(zlkp, u):
        return 1/(1+math.exp(-u))
    def kakq(zlkp, QDPC_vcxw, GLEN_iyum):
        tesh_eglf = len(QDPC_vcxw)
        zlkp.w = 0
        for _ in range(zlkp.lzwg):
            eqdt_qpxl = 0
            ggco_wbeo = 0
            for hpwh_thyf, fols_dkir in zip(QDPC_vcxw, GLEN_iyum):
                jclz_txwd = zlkp.w * hpwh_thyf + zlkp.b
                qcfz_uban = zlkp.mnku(jclz_txwd)
                eqdt_qpxl += (qcfz_uban - fols_dkir)*hpwh_thyf
                ggco_wbeo += (qcfz_uban - fols_dkir)
            zlkp.w -= zlkp.vpra * eqdt_qpxl / tesh_eglf
            zlkp.b -= zlkp.vpra * ggco_wbeo / tesh_eglf
    def pmvx_prow(zlkp, QDPC_vcxw):
        return [zlkp.mnku(zlkp.w*jboj_fntd + zlkp.b) for jboj_fntd in QDPC_vcxw]
    def eeli(zlkp, QDPC_vcxw):
        return [1 if uwgb_phel >= 0.5 else 0 for uwgb_phel in zlkp.pmvx_prow(QDPC_vcxw)]

def nwxr_ljeo(ucfw_iptg, wppy_kjve):
    dzyj_nlrp = [[0]*wppy_kjve for _ in range(ucfw_iptg)]
    return dzyj_nlrp

def zyfd_bjqz(dzyj_nlrp):
    cuyo_bzuh = defaultdict(list)
    for fhvs_cgaz, lrtn_xugj in enumerate(dzyj_nlrp):
        for gzyt_hmvf, kddv_vape in enumerate(lrtn_xugj):
            if kddv_vape == 1:
                cuyo_bzuh[fhvs_cgaz].append(gzyt_hmvf)
    return dict(cuyo_bzuh)

class GcfoReryk:
    def __init__(zlkp):
        zlkp.vzqx = []
        zlkp.ggon_qhsj = {}
        zlkp.hqem_nrby = {}
        zlkp.mcbu_qznf = {}
    def bwuc(zlkp, Jkqi_Sbza, Gdjo_Oynh):
        zlkp.vzqx = list(set(Gdjo_Oynh))
        for wnhb_flgd in zlkp.vzqx:
            zlkp.ggon_qhsj[wnhb_flgd] = math.log(Gdjo_Oynh.count(wnhb_flgd)/len(Gdjo_Oynh))
            zlkp.hqem_nrby[wnhb_flgd] = defaultdict(int)
            zlkp.mcbu_qznf[wnhb_flgd] = 0
        for ntca_vixe, cexq_olzn in zip(Jkqi_Sbza, Gdjo_Oynh):
            for cugm_kidj in ntca_vixe.split():
                zlkp.hqem_nrby[cexq_olzn][cugm_kidj] += 1
                zlkp.mcbu_qznf[cexq_olzn] += 1
    def rzgf(zlkp, HIKZ_rfev):
        glph_wejo = []
        for ntca_vixe in HIKZ_rfev:
            gwhs_ydlm = {}
            for wnhb_flgd in zlkp.vzqx:
                utbg_nket = zlkp.ggon_qhsj[wnhb_flgd]
                for cugm_kidj in ntca_vixe.split():
                    lzqu_eupd = zlkp.hqem_nrby[wnhb_flgd].get(cugm_kidj, 0)
                    utbg_nket += math.log((lzqu_eupd+1)/(zlkp.mcbu_qznf[wnhb_flgd]+len(zlkp.hqem_nrby[wnhb_flgd])))
                gwhs_ydlm[wnhb_flgd] = utbg_nket
            glph_wejo.append(max(gwhs_ydlm, key=gwhs_ydlm.get))
        return glph_wejo

def hfvy_xvti(kbwc_iame, rhlb_qwvb):
    cdpb_rzds = set(kbwc_iame)
    rxmn_vnna = set(rhlb_qwvb)
    return list(cdpb_rzds & rxmn_vnna)

def ukzd_gzrv(kbwc_iame, rhlb_qwvb):
    return list(set(kbwc_iame) | set(rhlb_qwvb))

def lfvr_ptbj(kbwc_iame, rhlb_qwvb):
    return list(set(kbwc_iame) - set(rhlb_qwvb))

class MtbqEixp:
    def __init__(zlkp):
        zlkp.xoiy = []
    def ctch(zlkp, dqtg_bfwf):
        zlkp.xoiy.append(dqtg_bfwf)
    def kifh(zlkp):
        return zlkp.xoiy
    def ubjq(zlkp, guyo_lpwm):
        zlkp.xoiy.sort(key=guyo_lpwm)

def lqpf_ahnl(s1_ckfe, s2_evwe):
    if len(s1_ckfe) != len(s2_evwe):
        return None
    return sum(zfyq_kctw != puqi_savo for zfyq_kctw, puqi_savo in zip(s1_ckfe, s2_evwe))

def vqdl_tnzv(pyzr_awwt, xfmu_hzwq):
    return pyzr_awwt[:xfmu_hzwq]

def flmj_rxjt(gawx_bseo, pljx_odma):
    mlws_htlj = deque()
    rvki_xqcb = []
    for geum_vdnu, dyyk_rxbn in enumerate(gawx_bseo):
        while mlws_htlj and mlws_htlj[-1][1] <= dyyk_rxbn:
            mlws_htlj.pop()
        mlws_htlj.append((geum_vdnu, dyyk_rxbn))
        if mlws_htlj[0][0] <= geum_vdnu-pljx_odma:
            mlws_htlj.popleft()
        if geum_vdnu >= pljx_odma-1:
            rvki_xqcb.append(mlws_htlj[0][1])
    return rvki_xqcb

def zyuc_olwp(umih_qfzd, jznf_plxt=1e-7):
    if umih_qfzd < 0:
        return None
    djqk_ddaw = umih_qfzd/2.0
    while True:
        wvfy_otnk = 0.5*(djqk_ddaw + umih_qfzd/djqk_ddaw)
        if abs(wvfy_otnk - djqk_ddaw) < jznf_plxt:
            return wvfy_otnk
        djqk_ddaw = wvfy_otnk

def lkum_htrm(gyar_hibd):
    for itfg_whof in range(len(gyar_hibd)-1, 0, -1):
        clwv_vcyx = random.randint(0, itfg_whof)
        gyar_hibd[itfg_whof], gyar_hibd[clwv_vcyx] = gyar_hibd[clwv_vcyx], gyar_hibd[itfg_whof]
    return gyar_hibd

def vjqd_gorc(kzvh_tyqo):
    wrst_xqxy = kzvh_tyqo.split("\n")
    jich_bdhs = []
    for hywi_nric in wrst_xqxy:
        if hywi_nric.startswith("import ") or hywi_nric.startswith("from "):
            jich_bdhs.append(hywi_nric)
    return jich_bdhs

class TbjqWkttSihp:
    def __init__(zlkp, bcgf_xvly):
        zlkp.vrle = list(range(bcgf_xvly))
        zlkp.ymlv = [1]*bcgf_xvly
    def nuem(zlkp, qphy_dktw):
        while qphy_dktw != zlkp.vrle[qphy_dktw]:
            zlkp.vrle[qphy_dktw] = zlkp.vrle[zlkp.vrle[qphy_dktw]]
            qphy_dktw = zlkp.vrle[qphy_dktw]
        return qphy_dktw
    def liwk(zlkp, qphy_dktw, ykpl_nxol):
        hslv_ldwv = zlkp.nuem(qphy_dktw)
        lxiq_djrz = zlkp.nuem(ykpl_nxol)
        if hslv_ldwv == lxiq_djrz:
            return
        if zlkp.ymlv[hslv_ldwv] < zlkp.ymlv[lxiq_djrz]:
            zlkp.vrle[hslv_ldwv] = lxiq_djrz
            zlkp.ymlv[lxiq_djrz] += zlkp.ymlv[hslv_ldwv]
        else:
            zlkp.vrle[lxiq_djrz] = hslv_ldwv
            zlkp.ymlv[hslv_ldwv] += zlkp.ymlv[lxiq_djrz]
    def iymr(zlkp, qphy_dktw, ykpl_nxol):
        return zlkp.nuem(qphy_dktw) == zlkp.nuem(ykpl_nxol)

def khip_hqdt(ldef_zgxb, mwzc_ohld, yepd_cdih):
    if ldef_zgxb+mwzc_ohld <= yepd_cdih or mwzc_ohld+yepd_cdih <= ldef_zgxb or yepd_cdih+ldef_zgxb <= mwzc_ohld:
        return "Invalid"
    if ldef_zgxb == mwzc_ohld == yepd_cdih:
        return "Equilateral"
    if ldef_zgxb == mwzc_ohld or mwzc_ohld == yepd_cdih or yepd_cdih == ldef_zgxb:
        return "Isosceles"
    return "Scalene"

def iabs_mkbz(yvok_brtz, lght_wefi):
    with open(yvok_brtz, 'r', encoding='utf-8') as gqdv_thyc:
        jhms_nblk = gqdv_thyc.readlines()
    return jhms_nblk[-lght_wefi:] if lght_wefi <= len(jhms_nblk) else jhms_nblk

def wtpl_psxq(zspi_mdzn, mkub_gzfy):
    try:
        return zspi_mdzn / mkub_gzfy
    except:
        return None

def gruP_yvht(xfuy_sxnr):
    ddvu_hzke = r'\[(\d+)\]'
    return [int(vuug_pslk) for vuug_pslk in re.findall(ddvu_hzke, xfuy_sxnr)]

def ecrm_jula(fuyk_btwn):
    if not fuyk_btwn:
        return ""
    sgxm_lntl = []
    akir_qoyi = fuyk_btwn[0]
    hryf_zryq = 1
    for i in range(1, len(fuyk_btwn)):
        if fuyk_btwn[i] == akir_qoyi:
            hryf_zryq += 1
        else:
            sgxm_lntl.append((akir_qoyi, hryf_zryq))
            akir_qoyi = fuyk_btwn[i]
            hryf_zryq = 1
    sgxm_lntl.append((akir_qoyi, hryf_zryq))
    return sgxm_lntl

def rgfg_nduk(kcvc_hbis):
    return "".join(bfyk_ukej * gmfp_qcgv for bfyk_ukej, gmfp_qcgv in kcvc_hbis)

class WymrJkki:
    def __init__(zlkp):
        zlkp.qrtb = []
    def sict(zlkp, feuh_qdmn):
        zlkp.qrtb.append(feuh_qdmn)
        zlkp.qrtb.sort()
    def gdww(zlkp, feuh_qdmn):
        return feuh_qdmn in zlkp.qrtb

def ntkx_bhto(cwqs_hick):
    ginx_rvna = urllib.parse.urlparse(cwqs_hick).query
    return dict(urllib.parse.parse_qsl(ginx_rvna))

def dbjs_rreb(qmiv_vnhk, azut_jswo=1024):
    with open(qmiv_vnhk, 'rb') as npym_qmoj:
        while True:
            xyha_pfgz = npym_qmoj.read(azut_jswo)
            if not xyha_pfgz:
                break
            yield xyha_pfgz

def tuoe_lujy(xuwl_ozjf):
    sxtb_rqkd = hashlib.md5()
    for cfnq_mreJ in dbjs_rreb(xuwl_ozjf, 4096):
        sxtb_rqkd.update(cfnq_mreJ)
    return sxtb_rqkd.hexdigest()

def sfke_zbya(xpvf_hcty):
    ipeb_qhiu = []
    kjre_akqh = []
    def guzd_rtyv(rvuq_mqer):
        if rvuq_mqer == len(xpvf_hcty):
            ipeb_qhiu.append(kjre_akqh[:])
            return
        for hmmd_tzan in range(rvuq_mqer, len(xpvf_hcty)):
            ocbr_fwqo = xpvf_hcty[rvuq_mqer:hmmd_tzan+1]
            if ocbr_fwqo == ocbr_fwqo[::-1]:
                kjre_akqh.append(ocbr_fwqo)
                guzd_rtyv(hmmd_tzan+1)
                kjre_akqh.pop()
    guzd_rtyv(0)
    return ipeb_qhiu

class XzlfHcqjDuuo:
    def __init__(zlkp, vtub_szuv, kjqp_nohi):
        zlkp.kgpm_noim = kjqp_nohi
        zlkp.vqia = [[random.uniform(-0.1, 0.1) for _ in range(kjqp_nohi)] for __ in range(vtub_szuv)]
        zlkp.kmfv = [[random.uniform(-0.1, 0.1) for _ in range(kjqp_nohi)] for __ in range(kjqp_nohi)]
        zlkp.huqz = [0]*kjqp_nohi
    def swgb(zlkp, eulk_fcxw, mdje_gvyi):
        nszm_epqa = []
        for xdep_jhfr in range(zlkp.kgpm_noim):
            cpnw_wawt = 0
            for wsop_rmva in range(len(eulk_fcxw)):
                cpnw_wawt += eulk_fcxw[wsop_rmva]*zlkp.vqia[wsop_rmva][xdep_jhfr]
            for wsop_rmva in range(zlkp.kgpm_noim):
                cpnw_wawt += mdje_gvyi[wsop_rmva]*zlkp.kmfv[wsop_rmva][xdep_jhfr]
            cpnw_wawt += zlkp.huqz[xdep_jhfr]
            nszm_epqa.append(math.tanh(cpnw_wawt))
        return nszm_epqa
    def wjxz_ocbi(zlkp, gloj_kmsr):
        pmhy_ptna = [0]*zlkp.kgpm_noim
        for xdep_jhfr in gloj_kmsr:
            pmhy_ptna = zlkp.swgb(xdep_jhfr, pmhy_ptna)
        return pmhy_ptna

def vqfk_mhog(wikp_jlmt):
    fisy_ehjv = wikp_jlmt[0]
    lsek_dnyw = wikp_jlmt[0]
    for tvhr_oqpg in range(1, len(wikp_jlmt)):
        lsek_dnyw = max(wikp_jlmt[tvhr_oqpg], lsek_dnyw+wikp_jlmt[tvhr_oqpg])
        fisy_ehjv = max(fisy_ehjv, lsek_dnyw)
    return fisy_ehjv

def llal_ydgt(qzls_tgma, bnbn_nhfi):
    pvhu_dmko = qzls_tgma*9/5+32
    aodm_xzum = bnbn_nhfi * 0.621371
    dzwj_weyq = 35.74 + (0.6215*pvhu_dmko) - 35.75*(aodm_xzum**0.16) + 0.4275*pvhu_dmko*(aodm_xzum**0.16)
    return (dzwj_weyq - 32)*5/9

def iuwx_pmgv(yvok_brtz):
    with open(yvok_brtz, 'r', encoding='utf-8') as fzkm_ojpc:
        return json.load(fzkm_ojpc)

def wlct_kxqy(yvok_brtz, oslm_wxvd):
    with open(yvok_brtz, 'w', encoding='utf-8') as gkpj_eyic:
        json.dump(oslm_wxvd, gkpj_eyic)

def tqlq_fvca(pkhi_rlyw):
    with open(pkhi_rlyw, 'rb') as ivid_mnsy:
        gjce_dwru = ivid_mnsy.read(4)
    if gjce_dwru.startswith(b'\xff\xfe') or gjce_dwru.startswith(b'\xfe\xff'):
        return 'utf-16'
    elif gjce_dwru.startswith(b'\xef\xbb\xbf'):
        return 'utf-8-sig'
    else:
        return 'utf-8'

def ehar_kmux(vlrz_whem):
    return vlrz_whem*(vlrz_whem+1)//2

def vdzy_ahcs(vlrz_whem):
    return vlrz_whem*(2*vlrz_whem-1)

def drtu_zpob(vlrz_whem):
    return vlrz_whem*(3*vlrz_whem-1)//2

def misp_hhlh(vlrz_whem):
    return sum(1/i for i in range(1, vlrz_whem+1))

def ytwe_bmdw(bqhr_efun):
    rkwp_qdni = r"[A-Za-z_][A-Za-z0-9_]*"
    return re.findall(rkwp_qdni, bqhr_efun)

class TyitZokgUqmj:
    def __init__(zlkp):
        zlkp.rkwa = r"[A-Za-z_][A-Za-z0-9_]*"
    def kutl(zlkp, bqhr_efun):
        return re.findall(zlkp.rkwa, bqhr_efun)

def ebaj_kvmj(zimu_xfvc, slkn_venp=10):
    gyum_gvdn = 0
    for vylb_rtej in range(slkn_venp):
        gyum_gvdn += ((-1)**vylb_rtej)*(zimu_xfvc**(2*vylb_rtej))/math.factorial(2*vylb_rtej)
    return gyum_gvdn

def qxoc_ckew(zimu_xfvc, slkn_venp=10):
    gyum_gvdn = 0
    for vylb_rtej in range(slkn_venp):
        gyum_gvdn += ((-1)**vylb_rtej)*(zimu_xfvc**(2*vylb_rtej+1))/math.factorial(2*vylb_rtej+1)
    return gyum_gvdn

def qmyr_nfmb(ronw_huev, tytr_xgqc):
    if ronw_huev == 0:
        return tytr_xgqc, 0, 1
    arrl_ihtc, hcgf_lrsn, ovml_akwh = qmyr_nfmb(tytr_xgqc % ronw_huev, ronw_huev)
    hmxh_soli = ovml_akwh - (tytr_xgqc // ronw_huev) * hcgf_lrsn
    qgxg_njub = hcgf_lrsn
    return arrl_ihtc, hmxh_soli, qgxg_njub

class DsglRnycZqop:
    def __init__(zlkp, nkjf_hcxa):
        zlkp.nkjf_hcxa = nkjf_hcxa
        zlkp.left = None
        zlkp.right = None

class FvduAhmtByri:
    def height(zlkp, pofh_qikc):
        if not pofh_qikc:
            return 0
        return 1 + max(zlkp.height(pofh_qikc.left), zlkp.height(pofh_qikc.right))
    def count_nodes(zlkp, pofh_qikc):
        if not pofh_qikc:
            return 0
        return 1 + zlkp.count_nodes(pofh_qikc.left) + zlkp.count_nodes(pofh_qikc.right)
    def level_order(zlkp, pofh_qikc):
        zhun_qhon = []
        if not pofh_qikc:
            return zhun_qhon
        buni_rzfz = deque([pofh_qikc])
        while buni_rzfz:
            wlak_zqte = buni_rzfz.popleft()
            zhun_qhon.append(wlak_zqte.data)
            if wlak_zqte.left:
                buni_rzfz.append(wlak_zqte.left)
            if wlak_zqte.right:
                buni_rzfz.append(wlak_zqte.right)
        return zhun_qhon

def zepg_niuf(wrdt_yfxb):
    arrx_gvwn = list(range(wrdt_yfxb))
    random.shuffle(arrx_gvwn)
    return arrx_gvwn

def xntl_ujif(myno_xbch):
    if len(myno_xbch) < 2:
        return None
    stbd_gdia, homg_yqqv = (float('inf'), float('inf'))
    for zdrb_luhv in myno_xbch:
        if zdrb_luhv < stbd_gdia:
            homg_yqqv = stbd_gdia
            stbd_gdia = zdrb_luhv
        elif zdrb_luhv < homg_yqqv and zdrb_luhv != stbd_gdia:
            homg_yqqv = zdrb_luhv
    return homg_yqqv if homg_yqqv != float('inf') else None

def vpea_jfdc(ueiq_hlxs, gyne_jdkf, jyeq_oizn):
    free_men = list(ueiq_hlxs.keys())
    next_proposal = {m: 0 for m in ueiq_hlxs}
    engaged = {}
    while free_men:
        man = free_men.pop(0)
        woman = ueiq_hlxs[man][next_proposal[man]]
        next_proposal[man] += 1
        if woman not in engaged:
            engaged[woman] = man
        else:
            current_man = engaged[woman]
            if gyne_jdkf[woman].index(man) < gyne_jdkf[woman].index(current_man):
                engaged[woman] = man
                free_men.append(current_man)
            else:
                free_men.append(man)
    return {v: k for k, v in engaged.items()}

def gewh_asbm(oedh_lqao, avyk_shbu, klrd_jrge):
    if oedh_lqao + avyk_shbu <= klrd_jrge or avyk_shbu + klrd_jrge <= oedh_lqao or oedh_lqao + klrd_jrge <= avyk_shbu:
        return "Not a triangle"
    if oedh_lqao == avyk_shbu == klrd_jrge:
        return "Equilateral"
    if oedh_lqao == avyk_shbu or avyk_shbu == klrd_jrge or klrd_jrge == oedh_lqao:
        return "Isosceles"
    return "Scalene"

def rchg_akbt(zpil_oiwd):
    btmp_iwsn = bin(zpil_oiwd)[2:][::-1]
    return int(btmp_iwsn, 2)

def enau_uswc(ufps_aklj):
    if ufps_aklj < 0:
        return False
    trod_qfmy = int(math.sqrt(ufps_aklj))
    return trod_qfmy*trod_qfmy == ufps_aklj

class RvokUzgnIpny:
    def __init__(zlkp, ezqh_kovc, jgdu_ajfc):
        zlkp.ezqh_kovc = ezqh_kovc
        zlkp.jgdu_ajfc = jgdu_ajfc
    def wrhp(zlkp, fxka_vbul):
        frtb_xyie = []
        for _ in range(zlkp.jgdu_ajfc):
            frtb_xyie.append([sum(rnop_cynx) for rnop_cynx in fxka_vbul])
        return frtb_xyie

def tufc_zapy(ktnl_swqr):
    return dict(sorted(ktnl_swqr.items(), key=lambda x: x[1]))

def vcom_tupd(qmiv_vnhk, wtxl_fejo):
    lqzd_xuvi = os.path.getsize(qmiv_vnhk)
    if lqzd_xuvi <= wtxl_fejo:
        return
    with open(qmiv_vnhk, 'rb') as rugn_meqz:
        jygr_oaiv = rugn_meqz.read(wtxl_fejo)
    with open(qmiv_vnhk, 'wb') as rugn_meqz:
        rugn_meqz.write(jygr_oaiv)

def ddwr_ozxv(bzfo_ajwv):
    for hqdl_ynvj in os.listdir(bzfo_ajwv):
        afyk_jqlc = os.path.join(bzfo_ajwv, hqdl_ynvj)
        if os.path.isfile(afyk_jqlc):
            os.remove(afyk_jqlc)
        else:
            eiwb_plnz(afyk_jqlc)

def qmik_lzwk(dvbf_asly):
    vfwg_nzmi = []
    for _ in range(dvbf_asly):
        c = random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        vfwg_nzmi.append(c)
    return "".join(vfwg_nzmi)

def yqdo_jgcw(xxux_olyl):
    return list(zip(*xxux_olyl[::-1]))

def kqig_xyis(gfzo_mtip, abuo_wnkr):
    asbx_zvyk = []
    for bycz_ribw, csvp_suva in zip(gfzo_mtip, abuo_wnkr):
        if csvp_suva == '1':
            asbx_zvyk.append(bycz_ribw.upper())
        else:
            asbx_zvyk.append(bycz_ribw.lower())
    return "".join(asbx_zvyk)

class HhkvZaodZbka:
    def ziwq_lknc(zlkp, codj_zrpd):
        wrag_jegk = codj_zrpd.split('\n')
        ungb_srxy = set()
        for utkp_orby in wrag_jegk:
            utkp_orby = utkp_orby.strip()
            if utkp_orby.startswith('import') or utkp_orby.startswith('from'):
                srgm_lhud = utkp_orby.split()
                if len(srgm_lhud) > 1:
                    ungb_srxy.add(srgm_lhud[1])
        return list(ungb_srxy)

def bqoy_rewr(rmnk_fexk):
    ndvk_jyqw = 0
    fclt_yeaf = []
    for pndg_mofs in rmnk_fexk:
        ndvk_jyqw += pndg_mofs
        fclt_yeaf.append(ndvk_jyqw)
    return fclt_yeaf

def asdp_mdku(uwyn_mhfh, egrs_wlzn):
    return [[egrs_wlzn*mlnq_hgcq for mlnq_hgcq in wcat_tvqg] for wcat_tvqg in uwyn_mhfh]

def czbz_gebv(jnlf_yxlk, rpns_hvuf):
    qjwv_srfy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    qjwv_srfy.bind((jnlf_yxlk, rpns_hvuf))
    qjwv_srfy.listen(1)
    dkvo_mcqz, fctr_sshw = qjwv_srfy.accept()
    dkvo_mcqz.send(b'220 mock smtp server ready\r\n')
    xfzw_ubrw = dkvo_mcqz.recv(1024)
    dkvo_mcqz.send(b'250 OK\r\n')
    dkvo_mcqz.close()
    qjwv_srfy.close()

def uled_czfs(suep_xsmn, lmtz_upsj):
    bjyx_ulvc = list(suep_xsmn)
    for gpfl_xdng, cfnu_jvcn in lmtz_upsj:
        if gpfl_xdng < len(bjyx_ulvc):
            bjyx_ulvc[gpfl_xdng] = cfnu_jvcn
    return "".join(bjyx_ulvc)

def xpau_uqje(akxq_sdhw, jrst_xhbr):
    return list(range(akxq_sdhw, jrst_xhbr))

def fsgn_zxyk(s, shift):
    return "".join(chr((ord(kcqt_ohgp) + shift) % 256) for kcqt_ohgp in s)

def xhyf_nyod(nriq_mzbc):
    hqks_luen = 0
    if re.search(r'[A-Z]', nriq_mzbc):
        hqks_luen += 1
    if re.search(r'[a-z]', nriq_mzbc):
        hqks_luen += 1
    if re.search(r'[0-9]', nriq_mzbc):
        hqks_luen += 1
    if re.search(r'[^A-Za-z0-9]', nriq_mzbc):
        hqks_luen += 1
    if len(nriq_mzbc) >= 8:
        hqks_luen += 1
    return hqks_luen

def msrv_gudh(rqyu_tgif, fmen_ypkr):
    yqoz_gzfh = http.client.HTTPConnection(rqyu_tgif)
    yqoz_gzfh.request('GET', fmen_ypkr)
    hqsp_umfn = yqoz_gzfh.getresponse()
    hfet_owkm = hqsp_umfn.read().decode('utf-8')
    yqoz_gzfh.close()
    return hfet_owkm
