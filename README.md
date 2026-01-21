# NifiETL Project

Proyek ini adalah implementasi pipeline ETL (Extract, Transform, Load) untuk data taksi NYC menggunakan Apache NiFi, Apache Spark, dengan monitoring menggunakan Prometheus dan Grafana.

## Deskripsi Proyek

Pipeline ETL ini dirancang untuk memproses data perjalanan taksi dari berbagai jenis layanan taksi di New York City:
- Yellow Taxi
- Green Taxi
- For-Hire Vehicle (FHV)
- For-Hire Vehicle High Volume (FHVHV)

Data diambil dari file CSV, diproses menggunakan Apache Spark untuk standardisasi dan transformasi, kemudian disimpan ke MongoDB. Sumber : https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page (Data asli parquet dikonversi ke CSV)

## Arsitektur

Proyek ini menggunakan arsitektur berbasis container dengan Docker Compose yang terdiri dari:

- **Apache NiFi**: Orchestrator untuk data flow dan pipeline management
- **Apache Spark**: Engine untuk ETL processing (Master + Worker)
- **Prometheus**: Monitoring dan alerting system
- **Grafana**: Dashboard untuk visualisasi metrics
- **Node Exporter**: Ekspor metrics sistem
- **StatsD Exporter**: Ekspor metrics aplikasi

## Teknologi yang Digunakan

- **Apache NiFi 1.27.0**
- **Apache Spark 3.5.6**
- **Prometheus**
- **Grafana**
- **MongoDB** (sebagai target penyimpanan)
- **Docker & Docker Compose**

## Struktur Proyek

```
NifiETL/
├── docker-compose.yml          # Konfigurasi container services
├── Dockerfile.nifi            # Dockerfile untuk Apache NiFi
├── data/                      # Folder untuk data input/output
├── spark/
│   ├── etl_job.py            # Script ETL utama
│   └── etl_bulk_job.py       # Script ETL untuk bulk processing
├── prometheus/
│   └── prometheus.yml        # Konfigurasi Prometheus
├── grafana/
│   └── provisioning/         # Konfigurasi Grafana
└── log spark.txt             # Log output dari Spark jobs
```

## Instalasi dan Setup

### Prasyarat

- Docker dan Docker Compose terinstall
- MongoDB instance (bisa lokal atau cloud)
- Minimal 8GB RAM untuk menjalankan semua services

### Langkah Instalasi

1. **Clone repository ini**
   ```bash
   git clone <repository-url>
   cd NifiETL
   ```

2. **Siapkan data input**
   - Letakkan file CSV data taksi di folder `data/`
   - File yang didukung: `fhv_tripdata_2025-*.csv`, `green_tripdata_2025-*.csv`, `yellow_tripdata_2025-*.csv`, `fhvhv_tripdata_2025-*.csv`

3. **Konfigurasi MongoDB**
   - Pastikan MongoDB berjalan dan dapat diakses
   - Update URI MongoDB di script ETL jika diperlukan

4. **Jalankan services**
   ```bash
   docker-compose up -build
   ```

5. **Import NIFI_Flow FINAL.json JSON ke Process Group**

## Cara Penggunaan

### Mengakses Services

- **NiFi Web UI**: https://localhost:8443 (default credentials: admin/admin)
- **Spark Master UI**: http://localhost:8081
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (default: admin/admin)

### Menjalankan ETL Job

ETL job dapat dijalankan melalui NiFi atau langsung via command line:

```bash
# Via Spark submit
docker exec spark-master-nifi /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/nifi/spark/etl_job.py \
  --input /data \
  --output mongodb://mongodb-host:27017/nyc_taxi.trips
```

### Monitoring

- **Prometheus**: Mengumpulkan metrics dari NiFi dan sistem
- **Grafana**: Visualisasi dashboard untuk monitoring pipeline performance

## Konfigurasi

### NiFi

- Port: 8443 (HTTPS)
- Data flow dapat dikonfigurasi melalui web UI
- Processor untuk Spark job submission sudah tersedia

### Spark

- Master: spark://spark-master:7077
- Worker: 6 cores, 6GB memory per worker
- Adaptive query execution enabled

### Prometheus

- Scrape interval: 15 detik
- Mengumpulkan metrics dari NiFi API dan node exporter

## Troubleshooting

### Common Issues

1. **Port conflicts**: Pastikan port 8443, 9090, 3000 tidak digunakan aplikasi lain
2. **Memory issues**: Jika Spark worker crash, tingkatkan memory limit di docker-compose.yml
3. **MongoDB connection**: Pastikan MongoDB dapat diakses dari container Spark


## Kontribusi

1. Fork repository
2. Buat branch fitur baru (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## Lisensi

Distributed under the MIT License. See `LICENSE` for more information.

## Kontak

- Project Link: [GitHub Repository URL]
- Email: [your-email@example.com]

---

**Catatan**: Proyek ini dibuat untuk keperluan akademik dan demonstrasi pipeline ETL modern menggunakan big data technologies.