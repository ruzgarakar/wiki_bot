# Wiki Araştırma Asistanı

Doğal dilde sorulan soruları ücretsiz bir yapay zeka dil modeliyle (Gemini, Groq, OpenRouter veya Ollama) anlayıp **Wikipedia API** üzerinden araştıran ve açıklayıcı bir özet sunan web sitesi.

## Nasıl çalışır?
1. Kullanıcı sorusunu kendi cümleleriyle yazar ve **Ara**'ya basar.
2. Dil modeli soruyu yorumlar, `wiki_search` aracıyla Wikipedia'da arama yapar, `wiki_read` ile ilgili makaleleri okur.
3. Okuduklarını sentezleyip Markdown biçimli, anlaşılır bir cevap yazar; kaynakları ve ilgili konuları listeler.
4. Arama/okuma adımları canlı olarak ekranda gösterilir.

## Özellikler
- Doğal dilde soru, takip soruları (sohbet bağlamı korunur)
- 7 Wikipedia dili (TR, EN, DE, FR, ES, AR, RU), 3 ayrıntı düzeyi
- Canlı araştırma adımları, resimli kaynak kartları, metin içi kaynak bağlantıları
- İlgili konu önerileri, "Daha basit anlat" / "Daha derine in"
- Kopyala, Markdown indir, yazdır/PDF, sesli okuma, sesle soru sorma
- Araştırma geçmişi (tarayıcıda saklanır), açık/koyu tema, mobil uyumlu, rastgele konu

## Kurulum
```bash
pip install -r requirements.txt
python app.py
```
Tarayıcıda http://localhost:5000 adresini açın.

## Ücretsiz dil modeli seçimi (`.env` dosyası)
| Sağlayıcı | `LLM_PROVIDER` | Ücretsiz anahtar | Not |
|---|---|---|---|
| Google Gemini (önerilen) | `gemini` | https://aistudio.google.com/apikey | Türkçesi çok iyi, Google hesabı yeterli |
| Groq | `groq` | https://console.groq.com/keys | Varsayılan model `openai/gpt-oss-120b`. Ücretsiz planda dakikada 8000 token sınırı olduğu için bir soru 30-60 sn sürebilir |
| OpenRouter | `openrouter` | https://openrouter.ai/keys | `:free` ile biten modeller ücretsiz |
| Ollama | `ollama` | gerekmez | Bilgisayarınızda çevrimdışı çalışır, güçlü bilgisayar ister |

Anahtarı `.env` içindeki `LLM_API_KEY=` satırına yazın. Modeli değiştirmek için `LLM_MODEL=` kullanın.
