"""GamePulse - Auditoria de aptitud de la capa plata de WoWAH (cobertura, silencios, vida, regimen, etiqueta y senales preliminares; se calcula sobre plata ya construida).
Uso (desde la raíz del repositorio): python analisis_adicional/auditoria_plata.py data/ <carpeta_salida_figuras> [zones.csv]
Lee los Parquet de plata (unica fuente del modelado) y, opcionalmente, zones.csv solo para enriquecer las senales preliminares (bloque 9c de la salida); no modifica datos. Requiere pandas, pyarrow, matplotlib."""
import sys, glob, numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow as pa
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SRC, OUT = sys.argv[1], sys.argv[2].rstrip('/') + '/'
ZONES = sys.argv[3] if len(sys.argv) > 3 else None
T0 = pd.Timestamp('2008-01-01'); D = lambda i: str(T0 + pd.Timedelta(days=int(i)))[:10]
B, OR, G = '#2B6CB0', '#D9730D', '#6B7280'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9CA3AF','axes.grid':True,'grid.color':'#E5E7EB','grid.linewidth':.6,'axes.axisbelow':True,'figure.dpi':150})
def h(t): print('\n=== ' + t + ' ===')

# 1. Reconciliacion -------------------------------------------------------------
h('1. Reconciliacion con el README')
fs = sorted(glob.glob(SRC + '/*.parquet'))
df = pa.concat_tables([pq.read_table(f) for f in fs]).to_pandas()
df['d'] = (df.ts.dt.normalize() - T0).dt.days
print('filas', len(df), '| personajes', df.char_id.nunique(), '| ts', df.ts.min(), '->', df.ts.max())
print('nulos por columna:', df.isna().sum().to_dict())
print('duplicados (char_id, ts):', int(df.duplicated(['char_id','ts']).sum()))
print('IDs inconsistentes:', df[df.id_inconsistente].char_id.nunique(), '| filas:', int(df.id_inconsistente.sum()))
print('gremio nulo: %d (%.2f%%)' % (df.gremio.isna().sum(), df.gremio.isna().mean()*100))
print('razas', df.raza.nunique(), '| clases', df.clase.nunique(), '| zonas', df.zona.nunique(), '| nivel', df.nivel.min(), '-', df.nivel.max())
ids = np.sort(df.char_id.unique()); ix = pd.Series(np.arange(len(ids)), index=ids)
bad = df.groupby('char_id').id_inconsistente.max().reindex(ids).values
M = np.zeros((len(ids), 366), bool); u = df[['char_id','d']].drop_duplicates(); M[ix[u.char_id].values, u.d.values] = True
print('personajes inconsistentes: %d (%.2f%% de los personajes)' % (bad.sum(), bad.mean()*100))

# 2. Cobertura diaria -----------------------------------------------------------
h('2. Cobertura temporal diaria')
reg = np.bincount(df.d, minlength=366); pers = M.sum(0); days = pd.date_range('2008-01-01','2008-12-31')
print('dias con datos', int((reg>0).sum()), 'de 366; dias SIN registros:', [D(i) for i in np.where(reg==0)[0]])
med = np.median(reg[reg>0]); print('mediana registros/dia', med)
low = [(D(i), int(reg[i]), int(pers[i])) for i in range(366) if 0 < reg[i] < .5*med]
print('dias con < 50%% de la mediana (%d):' % len(low), low)
m = df.groupby(df.ts.dt.month).agg(reg=('char_id','size')); m['pers_dia_prom'] = [pers[[i for i in range(366) if days[i].month==k]].mean().round(0) for k in range(1,13)]; print(m.to_string())
# 3. Cadencia --------------------------------------------------------------------
h('3. Cadencia entre observaciones consecutivas (min)')
dd = df.sort_values(['char_id','ts']); dt = dd.groupby('char_id').ts.diff().dt.total_seconds().dropna()/60
print(dt.describe(percentiles=[.1,.25,.5,.75,.9,.99]).round(2).to_string())
for th in [15,60,360,1440]: print('P(brecha > %d min) = %.4f' % (th, (dt>th).mean()))
# 4. Saltos entre dias activos ------------------------------------------------------
h('4. Saltos entre dias activos consecutivos (sin IDs inconsistentes)')
ok = dd[~dd.id_inconsistente][['char_id','d']].drop_duplicates(); s = ok.groupby('char_id').d.diff().dropna()
print('n saltos', len(s), '| salto == 1 dia:', int((s==1).sum()), '(%.1f%%)' % ((s==1).mean()*100))
for k in [2,3,7,14,21,30,45,60]: print('salto >= %2d d: %7d  (%.2f%%)' % (k, (s>=k).sum(), (s>=k).mean()*100))
f,ax=plt.subplots(figsize=(7,3.8)); ks=np.arange(1,91); ax.plot(ks,[(s>=k).mean()*100 for k in ks],color=B,lw=1.6); ax.set_yscale('log')
ax.axvline(30,color=OR,ls='--',lw=1); ax.text(31,ax.get_ylim()[1]*.6,'umbral 30 d: %.1f %%' % ((s>=30).mean()*100),color=OR,fontsize=9)
ax.set_xlabel('k (días entre dos días activos consecutivos del mismo personaje)'); ax.set_ylabel('% de saltos ≥ k (escala log)'); ax.set_title('Qué tan largos son los silencios entre apariciones',loc='left',fontsize=10); plt.tight_layout(); plt.savefig(OUT+'fig2_saltos_entre_dias_activos.png'); plt.close()
# 5. Vida del personaje ---------------------------------------------------------
h('5. Vida del personaje y composicion de la poblacion')
da = M.sum(1); first = M.argmax(1); last = 365 - M[:, ::-1].argmax(1)
print('dias activos: mediana %.0f, media %.1f, p25 %.0f, p75 %.0f' % (np.median(da), da.mean(), *np.percentile(da,[25,75])))
print('<=3 dias: %.3f | <=7: %.3f | >=60: %.3f' % ((da<=3).mean(), (da<=7).mean(), (da>=60).mean()))
fm = pd.Series(first).map(lambda i: days[i].month).value_counts().sort_index(); lm = pd.Series(last).map(lambda i: days[i].month).value_counts().sort_index()
print(pd.DataFrame({'primera_aparicion':fm,'ultima_aparicion':lm}).to_string())
f1 = dd.groupby('char_id').first(); print('primera obs con nivel<=5: %.3f | <=10: %.3f | primera obs en los 3 primeros dias del ano: %.3f' % ((f1.nivel<=5).mean(), (f1.nivel<=10).mean(), (first<=2).mean()))
oc = (first>=274)&(first<305); print('nuevos en octubre:', int(oc.sum()), '| con <=3 dias activos: %.3f | ultima aparicion en octubre: %.3f' % ((da[oc]<=3).mean(), (last[oc]<305).mean()))
print('nuevos/dia: mediana fuera de octubre', pd.Series(first)[~oc].value_counts().median(), '| 1-15 oct:', {i-273:int((first==i).sum()) for i in range(274,289)})
o = f1[(first>=280)&(first<=286)]; print('7-13 oct: n', len(o), '| nivel mediano 1a obs', o.nivel.median(), '| %% nivel<=5: %.2f' % (o.nivel<=5).mean(), '| zona:', o.zona.value_counts().head(3).to_dict())
f,ax=plt.subplots(1,2,figsize=(9,3.6)); vc=pd.Series(np.minimum(da,30)).value_counts().sort_index(); ax[0].bar(vc.index,vc.values,color=B,width=.8)
ax[0].set_xlabel('Días activos en 2008 (30 = 30 o más)'); ax[0].set_ylabel('Personajes'); ax[0].set_title('%.0f %% aparece ≤ 3 días' % ((da<=3).mean()*100),loc='left',fontsize=10)
x=np.arange(1,13); ax[1].plot(x,fm.reindex(x).values,color=B,marker='o',ms=4,label='Primera aparición'); ax[1].plot(x,lm.reindex(x).values,color=OR,marker='o',ms=4,label='Última aparición')
ax[1].set_xticks(x); ax[1].set_xlabel('Mes de 2008'); ax[1].set_title('Primera y última aparición por mes',loc='left',fontsize=10); ax[1].legend(frameon=False,fontsize=8); plt.tight_layout(); plt.savefig(OUT+'fig3_vida_personaje.png'); plt.close()
# 6. Regimen ----------------------------------------------------------------------
h('6. Cambio de regimen (expansion nominal 13/11/2008)')
exp = (pd.Timestamp('2008-11-13') - T0).days; dk = df[df.clase=='Death Knight']
print('primer Death Knight:', dk.ts.min(), '| filas', len(dk), '| personajes', dk.char_id.nunique()); print('primer registro con nivel > 70:', df[df.nivel>70].ts.min())
for a,b,l in [(0,exp,'antes 13/11'),(exp,366,'desde 13/11')]:
    sb = df[(df.d>=a)&(df.d<b)]; print(l, '| regs/dia %.0f | pers/dia %.0f | %%nivel70 %.3f | %%nivel80 %.3f | %%>70 %.3f' % (len(sb)/(b-a), sb.groupby('d').char_id.nunique().mean(), (sb.nivel==70).mean(), (sb.nivel==80).mean(), (sb.nivel>70).mean()))
pre = df[(df.d>=exp-14)&(df.d<exp)]; post = df[(df.d>=exp+1)&(df.d<=exp+14)]
print('14 d antes vs 14 d despues: regs/dia %.0f vs %.0f | pers/dia %.0f vs %.0f' % (len(pre)/14, len(post)/14, pre.groupby('d').char_id.nunique().mean(), post.groupby('d').char_id.nunique().mean()))
print('% registros en nivel 70 por mes:', df.groupby(df.ts.dt.month).nivel.apply(lambda v: round((v==70).mean(),3)).to_dict())
print('DK por dia (primeros):', dk.groupby(dk.d.map(D)).size().head(4).to_dict())
print('personajes/dia 12-nov..22-nov:', {D(i)[5:]:int(pers[i]) for i in range(316,327)})
lv=df.groupby('d').nivel.apply(lambda v:(v>70).mean()*100).reindex(range(366)).fillna(0).values; l70=df.groupby('d').nivel.apply(lambda v:(v==70).mean()*100).reindex(range(366)).values
f,ax=plt.subplots(figsize=(8,3.4)); ax.plot(days,l70,color=B,lw=1.4,label='% de registros en nivel 70'); ax.plot(days,lv,color=OR,lw=1.4,label='% de registros en nivel > 70')
ax.axvline(pd.Timestamp('2008-11-13'),color=G,ls='--',lw=.8); ax.legend(frameon=False,fontsize=8,loc='center left'); ax.set_ylabel('% de registros del día'); ax.set_title('Cambio de régimen: el tope de nivel',loc='left',fontsize=10); plt.tight_layout(); plt.savefig(OUT+'fig5_cambio_regimen.png'); plt.close()
f,ax=plt.subplots(2,1,figsize=(8,5),sharex=True); ax[0].plot(days,reg,color=B,lw=1); ax[0].set_ylabel('Registros por día'); ax[1].plot(days,pers,color=B,lw=1); ax[1].set_ylabel('Personajes por día')
for a in ax:
    for i in np.where(reg==0)[0]: a.axvspan(days[i],days[i]+pd.Timedelta(days=1),color=OR,alpha=.35,lw=0)
    for dd_ in ['2008-10-08','2008-11-13','2008-11-19']: a.axvline(pd.Timestamp(dd_),color=G,ls='--',lw=.8)
yl=ax[0].get_ylim()[1]
for dd_,l,ha,fy in [('2008-10-07','8-oct',  'right',.93),('2008-11-12','13-nov','right',.80),('2008-11-20','19-nov','left',.93)]: ax[0].text(pd.Timestamp(dd_),yl*fy,l,ha=ha,color=G,fontsize=8)
ax[0].set_title('Cobertura diaria de plata (naranja = días sin ningún registro)',loc='left',fontsize=10); plt.tight_layout(); plt.savefig(OUT+'fig1_cobertura_diaria.png'); plt.close()
# 7. Etiqueta por ventana -------------------------------------------------------------
h('7. Viabilidad de la etiqueta: corte c, historia = dias < c, futuro = dias [c, c+30)')
okm = ~bad; rows = []
for c in range(42, 336, 7):
    y = ~M[:, c:c+30].any(1); V = M[:, :c].any(1) & okm; A = M[:, c-14:c].any(1) & okm
    Bm = (M[:, c-28:c].sum(1) >= 3) & okm; C = Bm & (first <= c-14); r60 = M[:, c+30:c+90].any(1)
    rows.append(dict(corte=D(c), n_vistos=int(V.sum()), y_vistos=y[V].mean(), nA=int(A.sum()), yA=y[A].mean(), nB=int(Bm.sum()), yB=y[Bm].mean(), nC=int(C.sum()), yC=y[C].mean(),
                     pos_B=int((y&Bm).sum()), reaparece60_A=(r60[y&A].mean() if c+90<=366 and (y&A).sum() else np.nan)))
W = pd.DataFrame(rows); print(W.round(3).to_string(index=False)); W.to_csv(OUT+'ventanas_semanales.csv', index=False)
print('\nResumen: yB min/max %.3f/%.3f | nB min/max %d/%d | reaparece60_A min/max %.3f/%.3f' % (W.yB.min(), W.yB.max(), W.nB.min(), W.nB.max(), W.reaparece60_A.min(), W.reaparece60_A.max()))
c = 140; y = ~M[:, c:c+30].any(1); A = M[:, c-14:c].any(1); print('efecto 819 IDs (corte %s, activos 14 d): con %.4f | sin %.4f' % (D(c), y[A].mean(), y[A&okm].mean()))
print('ultimo corte con 30 d completos de seguimiento: %s (ventana hasta %s)' % (D(335), D(364)))
f,ax=plt.subplots(figsize=(8,3.9)); x=[days[c] for c in W.corte.map(lambda s:(pd.Timestamp(s)-T0).days)]
ax.plot(x,W.y_vistos*100,color=G,lw=1.4,label='Todos los vistos antes del corte'); ax.plot(x,W.yA*100,color=OR,lw=1.4,label='Activos en los 14 d previos'); ax.plot(x,W.yB*100,color=B,lw=1.8,label='≥ 3 días activos en los 28 d previos')
ax.set_ylabel('% sin registros en los 30 d siguientes'); ax.set_title('Tasa de la etiqueta según quién entra a la población (corte semanal)',loc='left',fontsize=10); ax.axvline(pd.Timestamp('2008-10-14'),color=G,ls='--',lw=.8); ax.legend(frameon=False,fontsize=8,loc='upper left'); plt.tight_layout(); plt.savefig(OUT+'fig4_tasa_etiqueta_por_poblacion.png'); plt.close()
# 8. Perfil descriptivo basico ------------------------------------------------------
h('8. Perfil descriptivo basico')
print('top zonas (% registros):', (df.zona.value_counts(normalize=True).head(5)*100).round(1).to_dict())
print('clase (% personajes):', (df.drop_duplicates('char_id').clase.value_counts(normalize=True)*100).round(1).to_dict())
print('raza (% personajes):', (df.drop_duplicates('char_id').raza.value_counts(normalize=True)*100).round(1).to_dict())
print('personajes con gremio alguna vez: %.3f' % df.groupby('char_id').gremio.apply(lambda v: v.notna().any()).mean())
ni = df.groupby('char_id').nivel.agg(['min','max']); print('personajes que suben >= 1 nivel en 2008: %.3f | llegan a nivel >= 70: %.3f' % (((ni['max']-ni['min'])>=1).mean(), (ni['max']>=70).mean()))

# 9. Senales preliminares (descriptivas, no causales) ---------------------------------
h('9. Senales preliminares sobre plata: tasa de y=1 por categoria de cada senal')
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d_ = 1 + z*z/n; c_ = (p + z*z/(2*n)) / d_; hw = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d_
    return (c_-hw, c_+hw)
dsort = df.sort_values(['char_id','ts']); okid = ~df.groupby('char_id').id_inconsistente.max().reindex(ids).values
CUTS = [84, 154, 252]   # 25-mar, 3-jun, 9-sep (antes del pico de octubre)
parts = []
for c in CUTS:
    hist = dsort[(dsort.d < c) & (dsort.d >= c-28)]
    elig = (M[:, c-28:c].sum(1) >= 3) & okid
    y = ~M[:, c:c+30].any(1)
    w14 = hist[hist.d >= c-14].groupby('char_id'); last = hist.groupby('char_id').last()
    f = pd.DataFrame(index=ids[elig]); f['y'] = y[elig].astype(int)
    f['niveles14'] = (w14.nivel.max() - w14.nivel.min()).reindex(f.index)
    f['nivel_ult'] = last.nivel.reindex(f.index)
    a14 = M[elig, c-14:c].sum(1); p14 = M[elig, c-28:c-14].sum(1); f['cambio_reg'] = a14 - p14
    f['zonas28'] = hist.groupby('char_id').zona.nunique().reindex(f.index)
    f['gremio'] = last.gremio.notna().reindex(f.index)
    f['zona_ult'] = last.zona.reindex(f.index)
    f['corte'] = D(c); parts.append(f)
S = pd.concat(parts); S = S[S.niveles14.notna()]
print('filas (personaje, corte):', len(S), '| personajes distintos:', S.index.nunique(), '| tasa y=1 global: %.3f' % S.y.mean())
for c in CUTS: print('  corte', D(c), 'n', int((S.corte==D(c)).sum()), 'tasa %.3f' % S[S.corte==D(c)].y.mean())
def tabla(nombre, cat):
    g = S.groupby(cat, observed=True).y.agg(['sum','count']); out = []
    for k, r in g.iterrows():
        lo, hi = wilson(r['sum'], r['count']); out.append((nombre, str(k), int(r['count']), r['sum']/r['count']*100, lo*100, hi*100))
    for o in out: print('%-22s %-14s n=%6d  y=1: %5.1f%%  IC95 [%4.1f, %4.1f]' % o)
    return out
S70 = S[S.nivel_ult < 70]
R = []
S['cap'] = np.where(S.nivel_ult >= 70, 'nivel 70+', 'menor a 70'); R += tabla('Tope de nivel', S.cap)
S['gr'] = np.where(S.gremio, 'con gremio', 'sin gremio'); R += tabla('Gremio (H4)', S.gr)
S['reg'] = pd.cut(S.cambio_reg, [-99,-3,-1,1,99], labels=['cae 3+ dias','cae 1-2','sin cambio','sube 2+']); R += tabla('Regularidad (H1)', S.reg)
S['zon'] = pd.cut(S.zonas28, [0,2,5,10,999], labels=['1-2 zonas','3-5','6-10','11+']); R += tabla('Variedad zonas (H3)', S.zon)
S70 = S[S.nivel_ult < 70].copy(); S70['prog'] = pd.cut(S70.niveles14, [-1,0,1,3,999], labels=['0 niveles','1','2-3','4+'])
g = S70.groupby('prog', observed=True).y.agg(['sum','count'])
for k, r in g.iterrows():
    lo, hi = wilson(r['sum'], r['count']); R.append(('Progreso 14d (H2)', str(k), int(r['count']), r['sum']/r['count']*100, lo*100, hi*100)); print('%-22s %-14s n=%6d  y=1: %5.1f%%  IC95 [%4.1f, %4.1f]' % R[-1])
# 9c. Enriquecimiento con zones.csv (solo analisis; no es parte de plata ni del modelado)
if ZONES:
    h('9c. Desajuste nivel-zona y tipo de contenido (zones.csv unido por nombre de zona)')
    Z = pd.read_csv(ZONES, na_values=['NA']).drop_duplicates('Zone_Name').set_index('Zone_Name')
    S['tipo'] = S.zona_ult.map(Z.Type); S['rmin'] = S.zona_ult.map(Z.Min_rec_level); S['rmax'] = S.zona_ult.map(Z.Max_rec_level)
    print('filas con zona unida: %.4f' % S.tipo.notna().mean())
    S['tipo_g'] = S.tipo.map({'City':'Ciudad','Zone':'Zona abierta','Dungeon':'Mazmorra/banda','Battleground':'Campo de batalla/arena','Arena':'Campo de batalla/arena'}).fillna('Otro'); R += tabla('Tipo de zona', S.tipo_g)
    V = S[(S.nivel_ult < 70) & S.tipo.isin(['Zone','Dungeon']) & (S.rmin >= 1) & (S.rmax >= S.rmin) & (S.rmax < 100)].copy()
    V['des'] = np.where(V.nivel_ult < V.rmin - 2, 'por debajo del minimo', np.where(V.nivel_ult > V.rmax + 2, 'por encima del maximo', 'dentro del rango'))
    print('filas con rango recomendado valido y nivel < 70: %d de %d' % (len(V), int((S.nivel_ult<70).sum())))
    g = V.groupby('des').y.agg(['sum','count'])
    for k in ['por debajo del minimo','dentro del rango','por encima del maximo']:
        r = g.loc[k]; lo, hi = wilson(r['sum'], r['count']); R.append(('Desajuste nivel-zona (H5)', k, int(r['count']), r['sum']/r['count']*100, lo*100, hi*100)); print('%-24s %-22s n=%6d  y=1: %5.1f%%  IC95 [%4.1f, %4.1f]' % R[-1])
# Ritmo temprano en personajes nuevos: historia = primeros 14 dias de vida; futuro = 30 d siguientes
h('9b. Personajes nuevos (primera obs. nivel<=5, aparicion antes de oct.): ritmo de nivel en sus primeros 14 d')
f1 = dsort.groupby('char_id').first(); fl = f1.nivel.reindex(ids).values; first_i = first
rows = []
for i in np.where((fl <= 5) & (first_i <= 273) & okid)[0]:
    s0 = first_i[i]; c = s0 + 14
    if M[i, s0:c].sum() < 3: continue
    rows.append((ids[i], s0, c, int(not M[i, c:c+30].any())))
N = pd.DataFrame(rows, columns=['id','s0','c','y']).set_index('id')
d14 = dsort[dsort.char_id.isin(N.index)].merge(N[['s0','c']], left_on='char_id', right_index=True); d14 = d14[(d14.d >= d14.s0) & (d14.d < d14.c)]
g14 = d14.groupby('char_id').nivel; N['niv'] = (g14.max() - g14.min()).reindex(N.index)
N['cat'] = pd.cut(N.niv, [-1,1,3,6,999], labels=['0-1 niveles','2-3','4-6','7+'])
print('nuevos elegibles:', len(N), '| tasa y=1 global %.3f' % N.y.mean())
gN = N.groupby('cat', observed=True).y.agg(['sum','count'])
for k, r in gN.iterrows():
    lo, hi = wilson(r['sum'], r['count']); R.append(('Ritmo inicial', str(k), int(r['count']), r['sum']/r['count']*100, lo*100, hi*100)); print('%-22s %-14s n=%6d  y=1: %5.1f%%  IC95 [%4.1f, %4.1f]' % R[-1])
# Figura 6
f, ax = plt.subplots(2, 4, figsize=(14, 6.2)); ax = ax.ravel()
paneles = [('Regularidad (H1)', 'Cambio de regularidad (14 d vs 14 d previos)'), ('Progreso 14d (H2)', 'Niveles ganados en 14 d (nivel < 70)'), ('Ritmo inicial', 'Personajes nuevos: niveles en sus primeros 14 d'),
           ('Variedad zonas (H3)', 'Zonas distintas en 28 d'), ('Gremio (H4)', 'Gremio'), ('Tope de nivel', 'Tope de nivel'), ('Desajuste nivel-zona (H5)', 'Nivel vs. rango de la zona (H5)'), ('Tipo de zona', 'Tipo de la última zona')]
for a, (k, t) in zip(ax, paneles):
    rr = [r for r in R if r[0] == k]; x = np.arange(len(rr)); v = np.array([r[3] for r in rr]); lo = np.array([r[4] for r in rr]); hi = np.array([r[5] for r in rr])
    a.bar(x, v, color=B, width=.6); a.errorbar(x, v, yerr=[v-lo, hi-v], fmt='none', ecolor='#111827', capsize=3, lw=1)
    a.set_xticks(x); a.set_xticklabels([r[1].replace(' ', '\n', 1 if len(r[1]) < 14 else 2) + '\n(n=%d)' % r[2] for r in rr], fontsize=7); a.set_title(t, loc='left', fontsize=9); a.set_ylabel('% con y=1')
plt.suptitle('Tasa de inactividad (y=1) por señal, con IC 95 % (Wilson). Descriptivo, no causal.', x=.01, ha='left', fontsize=10); plt.tight_layout(); plt.savefig(OUT+'fig6_senales_preliminares.png'); plt.close()
