<p align="center"><img src="figures/banner.png" width="100%"></p>

# atlas

> **manifold üzerinde az-etiketli öğrenme** — *(English: [README.md](README.md))*

**Sınıf başına bir etiket MNIST'i %94.7 doğrulukla sınıflandırır.** Etiketler zekice olduğu için değil — çünkü zor kısım hiçbir zaman etiket değildi. İyi bir temsili sıradan bir graf-difüzyon kuralına ver, on etiket yedi bin etiketin işini görür. Kötü bir temsil koy, *tıpatıp aynı kural* %71'e çöker.

<p align="center"><img src="figures/fig8_representation.png" width="70%"></p>

Aradaki 23 puanlık farkın tamamı **metriktir** — hangi noktaların "yakın" sayıldığı. Etiket ise geometrinin zaten bulduğu bir kümeye yalnızca isim koyar. Bu repo bunu kesinleştirir, ölçer ve tam olarak nerede kırıldığını gösterir.

<p align="center"><img src="figures/fig2_diffusion.gif" width="460"></p>
<p align="center"><em>İki etiketli nokta (yıldızlar). Renk graf boyunca akıp her ayı doldurur — boşluğu asla geçmez.</em></p>

### Sade anlatım

Binlerce görselin var, ama sadece bir avucunu etiketleyebiliyorsun. *Sınıflandırıcı eğitmek* yerine: (1) her görseli **etiketsiz** bir kodlayıcıyla öznitelik vektörüne çevir, (2) her görseli öznitelik uzayındaki en yakın komşularına bağlayıp graf kur, (3) **sınıf başına bir etiketli tohum** bırak ve her etiketin **graf kenarları boyunca yayılmasını** sağla — ta ki her düğüm renklenene dek. İnanılmaz iyi çalışır — *yeter ki* öznitelik uzayındaki komşular gerçekten aynı sınıftan olsun. İşte o tek koşul, **kenar saflığı**, her şeye karar verir ve bunu tamamen kodlayıcı belirler. Yani asıl soru hiçbir zaman "etiketleri nasıl kullanırız" değil — "metrik ne kadar iyi"dir.

---

## Üç iddia, her biri ölçülmüş

1. **Metrik her şeydir.** Aynı 10 etiket, aynı algoritma: ham piksel → %71, öğrenilmiş contrastive metrik → **%94.7** (tam-denetimli tavan: %98.6). Sınıflandırıcı değişmedi; geometri değişti.
2. **Etiketler sınıfları değil modları sayar.** Geometri veriyi grupladıktan sonra bir etiket, bir gruba verilen *isimden* ibarettir. Kabaca mod başına bir etiket gerekir ve sınıf, modların birleşimidir — yani etiket bütçesi kategori sayısını değil, verinin şeklini takip eder.
3. **Bu bir faz geçişidir, bir takas değil.** Keskin bir kenar-saflığı eşiği vardır. Üstünde tek etiket kümesini doğru boyar. Altında etiket yaymak *hiçbir şey yapmamaktan aktif olarak kötüdür*. Bu bir dipnot değil, taşıyıcı gerçek.

---

## Neden çalışır

**Öklid mesafesi manifoldda yalan söyler.** İki rakam piksel uzayında yakın ama farklı sınıflardan olabilir — çünkü aralarındaki düz çizgi iki katman arasındaki boşluğu keser. O metrikteki en-yakın-komşu, katmanların yaklaştığı her yerde yanılır.

<p align="center"><img src="figures/fig1_why_euclid_fails.png" width="80%"></p>
<p align="center"><em>Aynı nokta, aynı komşuluk büyüklüğü. Düz-çizgi topu diğer koldan 42 nokta kapar; graf komşuluğu 0 — manifoldu takip eder.</em></p>

**Graf metriği düzeltir.** Her noktayı k en yakın komşusuna bağla ve etiketin düz çizgide sıçraması yerine *kenarlar boyunca akmasına* izin ver. Artık "yakın" demek *veri boyunca ulaşılabilir* demektir. Sınıf başına tek tohumdan başlayan bu akış tüm veriyi boyar — yeter ki kenarlar çoğunlukla aynı sınıftan noktaları bağlasın. Her şeye karar veren tek sayı **kenar saflığıdır**.

---

## Mekanizma — matematiği

**Graf.** $n$ noktanın her birini bir $\phi$ temsiliyle göm, her noktayı $k$ en yakın komşusuna bağla, her kenarı Gauss çekirdeğiyle ağırlıklandır, simetrikleştir ($W=W^{\top}$) ve derece matrisi $D=\operatorname{diag}(\sum_j W_{ij})$ ile normalize et:

$$W_{ij} = \exp\!\left(-\frac{\lVert \phi_i-\phi_j\rVert^2}{2\sigma^2}\right),\qquad S = D^{-1/2}\,W\,D^{-1/2}.$$

**Etiket yayılımı.** Az sayıdaki etiketi bir one-hot tohum matrisi $Y$'ye koy (yalnız tohumlarda sıfırdan farklı) ve sabit noktaya kadar iterasyonla çalıştır:

$$F^{(t+1)} = \alpha\,S\,F^{(t)} + (1-\alpha)\,Y \;\;\xrightarrow{\;\alpha<1\;}\;\; F^{\star} = (1-\alpha)\,(I-\alpha S)^{-1}\,Y = (1-\alpha)\sum_{t=0}^{\infty}(\alpha S)^{t}\,Y,$$

sonra $\hat y_i=\arg\max_c F^{\star}_{ic}$ ile tahmin et. Asıl fikir bu seride: $(\alpha S)^{t}$, $t$-uzunluklu yürüyüş operatörüdür; yani bir noktanın etiketi ona **her uzunluktaki yürüyüşlerle ulaşan etiket kütlesinin iskontolu toplamıdır** — etiket kenarlar boyunca akar, kısa yürüyüşler en ağır. Eşdeğer olarak $F^\star$, şunun tek minimize edicisidir:

$$\tfrac12\sum_{i,j} W_{ij}\left\lVert \tfrac{F_i}{\sqrt{D_{ii}}}-\tfrac{F_j}{\sqrt{D_{jj}}}\right\rVert^2 \;+\; \tfrac{1-\alpha}{\alpha}\sum_i \lVert F_i-Y_i\rVert^2,$$

güçlü kenarlar boyunca yavaş değişen, tohumlara yakın kalan etiketler.

**Neden her şey saflıkta.** $S=S_{\text{iç}}+S_{\text{çapraz}}$ diye ayır (sınıf-içi ve çapraz kenarlar). Yürüyüş serisi tohumun etiketini $S_{\text{iç}}$ üzerinden doğru taşır; $S_{\text{çapraz}}$ içeren her terim, *yanlış* etiketi sınırın karşısına sızdıran bir kanaldır. Kenar saflığı $\pi=\tfrac{\text{sınıf-içi kenar ağırlığı}}{\text{toplam}}$ ile: $\pi\to1$ iken $S$ sınıflara göre blok-köşegen olur, difüzyon evinde kalır; $\pi$ düşerken çapraz iletkenlik büyür ve bir eşiği geçince bir düğümdeki yanlış-etiket kütlesi doğruyu aşar, $\arg\max$ **döner**. Uçurum, yokuş değil.

**$\pi$'yi yükselten temsil.** Contrastive kodlayıcı, **etiketsiz** eğitilerek, bir görüntünün iki artırımını yakınlaştırır, geri kalan her şeyi iter (NT-Xent, sıcaklık $\tau$):

$$\mathcal{L}_i = -\log\frac{\exp(\langle z_i,z_i^{+}\rangle/\tau)}{\sum_{j\ne i}\exp(\langle z_i,z_j\rangle/\tau)}.$$

Bu, metriği aynı-sınıf görüntüler komşu olacak şekilde büker; kenar saflığı %88'den (ham) %97'ye (contrastive) fırlar — faz geçişinin tam da ödüllendirdiği şey.

---

## Gerçek kılan yakalama: faz geçişi

Metriği boz, kenar saflığı düşer. Graf difüzyonu saflığı **dik** takip eder; naif 1-NN neredeyse kıpırdamaz. Bir kesişimin altında (MNIST'te ~%65 saflık) "akıllı" yöntem *temelden daha kötüdür* — tek bir ağır "köprü" kenarı koca bir bölgeyi yanlış etiketle sular.

<p align="center">
  <img src="figures/fig4_phase_transition.png" width="49%">
  <img src="figures/fig3_bridge_edge.png" width="49%">
</p>
<p align="center"><em>Solda: ~%65 saflığın altında difüzyon (turuncu) naif 1-NN'in (gri) altına iner — uçurum, yokuş değil. Sağda: tek köprü kenarı bir sınırı sular, %100 → %82.</em></p>

Sonucun işe yarar kısmı bu: *ne zaman uğraşmayacağını* söyler. Temsilin düşük-saflıklı bir graf veriyorsa, etiket yayılımı sana etiket kazandırmaz, doğruluk kaybettirir.

---

## Bir veri kümesi gerçekte kaç etiket ister?

Temsili $N$ kümeye ayır ve her birine bir etiket ver. $N$ büyüdükçe doğruluk küme-başı çoğunluk tavanına doğru tırmanır — doğruluğu *isimlerle* satın alıyorsun ve para birimi **modlar, sınıflar değil**. On etiket (sınıf başına bir), bu eğrinin sadece en kaba noktası.

<p align="center"><img src="figures/fig5_label_budget.png" width="64%"></p>
<p align="center"><em>Daha çok küme = daha çok isim = daha çok doğruluk, küme-başı tavana kadar. x-ekseni etiket bütçen; para birimi modlar.</em></p>

---

## On etiket, haritanın üzerinde

Contrastive gömme; solda gerçek etiketlerle, sağda yalnızca 10 tohumdan difüze edilen etiketle (siyah yıldızlar; hatalar kırmızı). On isim, bütün harita.

<p align="center"><img src="figures/fig6_embedding.png" width="88%"></p>
<p align="center"><em>Solda: gerçek rakamla renklendirilmiş gömme. Sağda: 10 tohumdan (yıldızlar) difüze edilen etiket; kırmızı = sınırlardaki %4 hata.</em></p>

---

## Yeniden üretim

```bash
pip install -r requirements.txt
python src/evaluate.py          # tablo               -> results/results.json
python src/sweep.py             # saflık + bütçe      -> results/sweep.json
python figures/make_figures.py  # tüm figürler (kapak GIF'i dahil)
```

Contrastive gömme ve değerlendirme bölmesi repoda gelir; her sayı **GPU olmadan** yeniden üretilir (MNIST ilk kullanımda kendini indirir). Kodlayıcıyı yeniden eğit: `pip install torch && python src/train_contrastive.py 40` (MPS / CUDA / CPU otomatik). Tam Mac/NVIDIA kurulumu: [`docs/TRAINING.md`](docs/TRAINING.md). 5 hücrelik tur: [`notebooks/demo.ipynb`](notebooks/demo.ipynb).

---

## MNIST sonuçları

Contrastive metrikte graf difüzyonu 10 etiketle **%94.7** — hepsi 7000 etiketle eğitilmiş tam-denetimli modelin sadece 3.9 puan altında, **700× daha az etiketle**.

<p align="center"><img src="figures/fig7_headline.png" width="60%"></p>

**Sayılar** (MNIST, 10k, sınıf başına 1 etiket, 60 rastgele seçim, k=10) — doğruluk kaynağı `results/*.json`:

| temsil | kenar saflığı | Öklid 1-NN | graf difüzyonu |
|---|--:|--:|--:|
| ham piksel | %88.4 | %42.3 | %71.4 |
| PCA-50 | %89.9 | %43.6 | %74.6 |
| difüzyon haritası | %91.3 | %37.6 | %72.8 |
| **contrastive** | **%96.6** | **%62.9** | **%94.7** |

Son iki sütuna bak: graf difüzyonu naif 1-NN'i *yalnızca* yüksek-saflıklı contrastive grafta ezer (94.7'ye 62.9) — ve fark yaratan satır, contrastive, bir etiket numarası değil temsildir.

---

## Gerçek fotoğraflarda tutuyor mu? — ImageNette

MNIST kolaydır. O yüzden dürüst test gerçek ImageNet fotoğrafları: [ImageNette](https://github.com/fastai/imagenette) (10 sınıf — tench, church, parachute, …), sıfırdan **etiketsiz** eğitilmiş bir ResNet-18 contrastive kodlayıcı (160px, 300 epoch), sonra 3925 görüntülük doğrulama havuzunda aynı sınıf-başına-1-etiket değerlendirmesi.

**Tez daha da güçleniyor.** Gerçek görüntülerde piksel metriği neredeyse işe yaramaz — ham-piksel difüzyonu şans seviyesinde (%13). Öğrenilmiş metrik aynı 10 etiketle **%62'ye** sıçrar. Temsil farkı MNIST'teki +23 puandan burada **+49** puana çıkar.

<p align="center"><img src="figures/imagenette_representation.png" width="70%"></p>

| temsil | kenar saflığı | Öklid 1-NN | graf difüzyonu |
|---|--:|--:|--:|
| ham piksel | %21.1 | %15.4 | %13.3 |
| PCA-50 | %22.8 | %15.8 | %14.4 |
| difüzyon haritası | %21.5 | %13.1 | %15.1 |
| **contrastive (ResNet-18, 300 ep)** | **%75.3** | **%49.5** | **%61.9** |

**Ve faz geçişi aynı yerde.** Contrastive metriği gürültüyle boz, difüzyon naif 1-NN'in altına **~%63 kenar saflığında** iner — neredeyse MNIST eşiği (~%65). Saflık uçurumu veri kümesinin değil, *yöntemin* bir özelliği gibi.

<p align="center">
  <img src="figures/imagenette_phase.png" width="49%">
  <img src="figures/imagenette_embedding.png" width="49%">
</p>

**Dürüst kapsam.** Az etiket **%61.9**'a ulaşır, tam-denetimli tavan **%83.3** (tüm ~9500 train etiketiyle lineer prob) — tavanın ~%74'ü, hâlâ MNIST'in ~%96'sının altında. Kalan boşluk etiketlerin değil, *temsilin* suçu: 9k görüntüde sıfırdan ResNet-18 hâlâ mütevazı bir metriktir; daha ağır bir kodlayıcı (ResNet-50, daha çok epoch, büyük batch — bkz. [`docs/TRAINING.md`](docs/TRAINING.md)) hem tavanı hem ulaşılan oranı yükseltir. Bütün mesele bu, yeniden ifade edilmiş hâli.

### MNIST'e karşı ImageNette (özet)

| | MNIST | ImageNette |
|---|--:|--:|
| ham-piksel (10 etiket) | %71.4 | %13.3 |
| contrastive (10 etiket) | %94.7 | %61.9 |
| **temsil farkı** | **+23 puan** | **+49 puan** |
| tam-etiket tavanı | %98.6 | %83.3 |
| tavanın yakalanan oranı | ~%96 | ~%74 |
| faz-geçişi eşiği | ~%65 saflık | ~%63 saflık |

```bash
python src/train_contrastive_imagenette.py --epochs 300 --img 160 --batch 512   # hızlı GPU'da ~40 dk
python src/evaluate.py --dataset imagenette
python src/sweep.py --dataset imagenette && python figures/make_imagenette_figures.py
```

Eğitilmiş gömme (`data/imagenette_emb.npz`) repoda gelir; tablo ve figürler **GPU olmadan** yeniden üretilir. **Mac (MPS) ve NVIDIA (CUDA, RTX 50-serisi dahil) için tam kurulum — torch cu128 kurulumu, VRAM rehberi, önerilen config'ler, AMP — [`docs/TRAINING.md`](docs/TRAINING.md)'de** (cihaz otomatik seçilir; CUDA'da mixed-precision açık; daha çok kapasite için `--backbone resnet50`).

---

## Daha fazlası — daha iyi cetvel mi, daha iyi seçilmiş etiket mi?

İki doğal yükseltme, her biri dürüst ablasyon olarak ölçüldü (`src/enhance.py`, shipped gömmeler üzerinde):

1. **Riemann metriği** — tek global mesafe ölçeği yerine *yerel-uyarlanır, anizotropik* cetvel: self-tuning per-nokta ölçek, Coifman α=1 normalizasyonu (**Laplace–Beltrami** operatörünü = manifoldun içsel geometrisini geri verir), ve **yerel ters-kovaryans metrik tensörü** g(x) (Mahalanobis; "her yönün kendi ağırlığı").
2. **Hangi noktayı etiketleyeceğini seçmek** — her örnek eşit iyi bir tohum değil. Sınıf başına **en tipik** noktayı (medoid / yoğunluk-tepesi) etiketle, rastgele değil.

<p align="center"><img src="figures/fig9_enhance.png" width="82%"></p>

**Sayılar ne diyor (dürüstçe):**

- **İyi bir temsilde daha iyi metrik neredeyse hiç yardım etmiyor.** Contrastive'de self-tuning ve cosine düz, α=1 ~1 puan bile kaybettiriyor — encoder aynı-sınıf noktaları zaten yan yana koymuş, mesafeyi yeniden ağırlıklandırmanın düzeltecek pek şeyi kalmamış. Yerel-Mahalanobis (Riemann) metriği tam da cetvelin kötü olduğu yerde küçük *gerçek* kazanç veriyor: **ImageNette'te +1.8 puan** (zayıf metrik), MNIST'te ~0. Daha iyi cetvel yalnız cetvel kötüyken işe yarar.
- **Asıl kaldıraç: hangi etiketi seçtiğin.** En tipik noktayı (rastgele yerine) tohumlamak **MNIST 94.7 → 97.0 (+2.3)** ve **ImageNette 61.9 → 78.5 (+16.6)** — neredeyse %83.3 tavanına. Düşük-saflıklı grafta rastgele tohum çoğu zaman atipik, sınırdaki bir görsele düşüp yanlış bölgeyi sular; prototip çekirdekte oturup temiz yayılır. Buradaki tipiklik **etiketsiz** ölçülür (graf derecesi / medoid), yani adil bir az-etiket hamlesi — ve tam olarak aktif öğrenme.

Çıkarım: temsil iyi olduğunda marjinal kazanç, hesapladığın mesafede değil, **seçtiğin etikettedir**. (Başlık tablosu karşılaştırılabilirlik için kötümser *rastgele* tohumu tutar; bu, pratikte kullanacağın ek.)

---

## Bu ne

Klasik bir primitifin temiz, dürüst bir **ölçümü** — yeni bir algoritma değil: etiket yayılımı Zhou ve ark. (2004), difüzyon haritaları Coifman & Lafon (2006). atlas'ın kattığı şey, iki okunabilir veri kümesinde anatomisi — kenar-saflığı faz geçişi (MNIST *ve* ImageNette'te aynı ~%60-65 eşik), temsil-her-şeydir farkı ve modlar-sınıflar-değil etiket yasası — her biri tek komutla üretilir. MIT lisanslı.
