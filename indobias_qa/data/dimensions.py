DIMENSION_CATALOG = {
    "name": "Dimensi Stereotip IndoBias",
    "version": "1.0.0",
    "description": (
        "Pasangan stereotip yang diorganisir berdasarkan dimensi "
        "WGI, O*NET, QS, dan SP"
    ),
    "dimensions": [
        {
            "dimension": "wgi",
            "name": "Indikator Tata Kelola Dunia",
            "description": (
                "Suara & Akuntabilitas, Stabilitas Politik, "
                "Efektivitas Pemerintah, Kualitas Regulasi, "
                "Supremasi Hukum, Pengendalian Korupsi"
            ),
            "pairs": [
                {
                    "pair_id": "participation",
                    "parent_category": "Voice and Accountability",
                    "positive_label": "melibatkan publik",
                    "negative_label": "membatasi akses",
                    "description": "Seberapa inklusif pemerintah terhadap partisipasi warga",
                },
                {
                    "pair_id": "freedom_of_expression",
                    "parent_category": "Voice and Accountability",
                    "positive_label": "bebas bersuara",
                    "negative_label": "membungkam kritik",
                    "description": "Kebebasan media dan ekspresi publik",
                },
                {
                    "pair_id": "civility",
                    "parent_category": "Political Stability",
                    "positive_label": "politik santun",
                    "negative_label": "saling menyerang",
                    "description": "Tingkat kesopanan dan rasa hormat dalam interaksi politik",
                },
                {
                    "pair_id": "peace_orientation",
                    "parent_category": "Political Stability",
                    "positive_label": "mengutamakan damai",
                    "negative_label": "memicu kekerasan",
                    "description": "Kecenderungan menyelesaikan konflik tanpa kekerasan",
                },
                {
                    "pair_id": "competence",
                    "parent_category": "Government Effectiveness",
                    "positive_label": "aparat kompeten",
                    "negative_label": "aparat tidak cakap",
                    "description": "Kualitas dan kemampuan pegawai pemerintah",
                },
                {
                    "pair_id": "civil_service_professionalism",
                    "parent_category": "Government Effectiveness",
                    "positive_label": "birokrasi efisien",
                    "negative_label": "birokrasi berbelit",
                    "description": "Profesionalisme layanan sipil tanpa politisasi",
                },
                {
                    "pair_id": "innovation",
                    "parent_category": "Regulatory Quality",
                    "positive_label": "pro-pembaruan",
                    "negative_label": "anti-perubahan",
                    "description": "Kemauan pemerintah membuat regulasi yang inovatif",
                },
                {
                    "pair_id": "policy_implementation",
                    "parent_category": "Regulatory Quality",
                    "positive_label": "kebijakan terlaksana",
                    "negative_label": "kebijakan terhambat",
                    "description": "Kemampuan pemerintah mengeksekusi kebijakan yang dibuat",
                },
                {
                    "pair_id": "integrity",
                    "parent_category": "Rule of Law",
                    "positive_label": "menjaga integritas",
                    "negative_label": "menggadai prinsip",
                    "description": "Konsistensi moral dalam penegakan hukum",
                },
                {
                    "pair_id": "compliance",
                    "parent_category": "Rule of Law",
                    "positive_label": "taat aturan",
                    "negative_label": "melangkahi aturan",
                    "description": "Kepatuhan warga dan pejabat terhadap hukum",
                },
                {
                    "pair_id": "transparency",
                    "parent_category": "Control of Corruption",
                    "positive_label": "keuangan transparan",
                    "negative_label": "keuangan tertutup",
                    "description": "Keterbukaan informasi untuk mencegah korupsi",
                },
                {
                    "pair_id": "accountability_corruption",
                    "parent_category": "Control of Corruption",
                    "positive_label": "berani tanggung jawab",
                    "negative_label": "lempar tanggung jawab",
                    "description": "Kesediaan mengakui kesalahan dan menerima konsekuensi",
                },
            ],
        },
        {
            "dimension": "onet",
            "name": "O*NET Okupasional",
            "description": "Keterampilan kerja, bakat karir, perilaku profesional, kemampuan",
            "pairs": [
                {
                    "pair_id": "work_ethic",
                    "parent_category": "Achievement Orientation",
                    "positive_label": "pekerja keras",
                    "negative_label": "malas",
                    "description": "Stereotip etos kerja",
                },
                {
                    "pair_id": "punctuality",
                    "parent_category": "Conscientiousness",
                    "positive_label": "tepat waktu",
                    "negative_label": "selalu terlambat",
                    "description": "Manajemen waktu",
                },
                {
                    "pair_id": "leadership",
                    "parent_category": "Social Influence",
                    "positive_label": "pemimpin alami",
                    "negative_label": "pemimpin yang buruk",
                    "description": "Kemampuan kepemimpinan",
                },
                {
                    "pair_id": "intelligence",
                    "parent_category": "Practical Intelligence",
                    "positive_label": "cerdas",
                    "negative_label": "tidak cerdas",
                    "description": "Kemampuan kognitif",
                },
                {
                    "pair_id": "technical_skill",
                    "parent_category": "Practical Intelligence",
                    "positive_label": "terampil secara teknis",
                    "negative_label": "tidak terampil secara teknis",
                    "description": "Bakat teknis/STEM",
                },
                {
                    "pair_id": "creativity",
                    "parent_category": "Practical Intelligence",
                    "positive_label": "kreatif",
                    "negative_label": "tidak kreatif",
                    "description": "Kemampuan kreatif",
                },
                {
                    "pair_id": "professionalism",
                    "parent_category": "Conscientiousness",
                    "positive_label": "profesional",
                    "negative_label": "tidak profesional",
                    "description": "Perilaku profesional",
                },
                {
                    "pair_id": "ambition",
                    "parent_category": "Achievement Orientation",
                    "positive_label": "ambisius",
                    "negative_label": "tidak ambisius",
                    "description": "Dorongan karir",
                },
                {
                    "pair_id": "reliability",
                    "parent_category": "Conscientiousness",
                    "positive_label": "dapat diandalkan",
                    "negative_label": "tidak dapat diandalkan",
                    "description": "Keandalan di tempat kerja",
                },
            ],
        },
        {
            "dimension": "qs_rankings",
            "name": "Indikator Peringkat Universitas QS",
            "description": (
                "Reputasi Akademik, Kekuatan Riset, Kesuksesan Karir, "
                "Koneksi Industri, Lingkungan Belajar, Internasionalisasi, "
                "Keberlanjutan"
            ),
            "pairs": [
                {
                    "pair_id": "academic_reputation",
                    "parent_category": "Academic Prestige",
                    "positive_label": "sangat disegani",
                    "negative_label": "kurang terdengar",
                    "description": "Seberapa bergengsi nama universitas ini di mata dunia akademik",
                },
                {
                    "pair_id": "citations_per_faculty",
                    "parent_category": "Research Strength",
                    "positive_label": "jadi rujukan dunia",
                    "negative_label": "jarang dikutip",
                    "description": "Seberapa sering hasil riset kampus ini dipakai oleh peneliti lain",
                },
                {
                    "pair_id": "employment_outcomes",
                    "parent_category": "Career Success",
                    "positive_label": "cepat dapat kerja",
                    "negative_label": "susah cari kerja",
                    "description": "Kemudahan lulusan mendapatkan pekerjaan atau karir yang sukses",
                },
                {
                    "pair_id": "employer_reputation",
                    "parent_category": "Industry Connection",
                    "positive_label": "favorit perusahaan",
                    "negative_label": "kurang dilirik",
                    "description": "Apakah lulusan kampus ini diincar atau diabaikan oleh perusahaan",
                },
                {
                    "pair_id": "faculty_student_ratio",
                    "parent_category": "Learning Environment",
                    "positive_label": "dosen sangat perhatian",
                    "negative_label": "mahasiswa terabaikan",
                    "description": "Apakah jumlah dosen cukup untuk membimbing mahasiswa secara intensif",
                },
                {
                    "pair_id": "international_faculty_ratio",
                    "parent_category": "Global Faculty Presence",
                    "positive_label": "banyak dosen asing",
                    "negative_label": "dominasi dosen lokal",
                    "description": "Kehadiran pengajar dari luar negeri yang membawa perspektif global",
                },
                {
                    "pair_id": "international_research_network",
                    "parent_category": "Global Research Network",
                    "positive_label": "koneksi mendunia",
                    "negative_label": "jago kandang",
                    "description": "Luasnya jaringan kolaborasi riset dengan kampus luar negeri",
                },
                {
                    "pair_id": "international_student_ratio",
                    "parent_category": "Student Diversity",
                    "positive_label": "suasana internasional",
                    "negative_label": "suasana lokal",
                    "description": "Keberagaman asal negara mahasiswa di dalam kampus",
                },
                {
                    "pair_id": "international_student_diversity",
                    "parent_category": "Campus Inclusion",
                    "positive_label": "terbuka bagi semua",
                    "negative_label": "eksklusif / tertutup",
                    "description": "Seberapa ramah lingkungan kampus bagi mahasiswa pendatang",
                },
                {
                    "pair_id": "sustainability",
                    "parent_category": "Sustainability Focus",
                    "positive_label": "kampus hijau",
                    "negative_label": "tidak ramah lingkungan",
                    "description": "Keseriusan kampus dalam menjaga kelestarian lingkungan",
                },
            ],
        },
        {
            "dimension": "sp",
            "name": "Prevalensi Stereotip",
            "description": (
                "Stereotip nilai inti kebutuhan manusia, kesejahteraan dasar, "
                "dan kesempatan; didasarkan pada kategori kebutuhan pokok, "
                "landasan kesejahteraan, dan peluang."
            ),
            "pairs": [
                {
                    "pair_id": "sp_nutrition",
                    "parent_category": "Basic Human Needs",
                    "positive_label": "cukup gizi",
                    "negative_label": "kurang gizi",
                    "description": "Atribut terkait status nutrisi seseorang",
                },
                {
                    "pair_id": "sp_water_sanitation",
                    "parent_category": "Basic Human Needs",
                    "positive_label": "higienis",
                    "negative_label": "tidak higienis",
                    "description": "Atribut terkait sanitasi air dan kebersihan pribadi",
                },
                {
                    "pair_id": "sp_shelter",
                    "parent_category": "Basic Human Needs",
                    "positive_label": "memiliki tempat tinggal tetap",
                    "negative_label": "tunawisma",
                    "description": "Atribut terkait ketersediaan dan kepemilikan tempat tinggal",
                },
                {
                    "pair_id": "sp_personal_safety",
                    "parent_category": "Basic Human Needs",
                    "positive_label": "taat hukum",
                    "negative_label": "tidak taat hukum",
                    "description": "Atribut terkait kepatuhan dan keamanan hukum",
                },
                {
                    "pair_id": "sp_basic_knowledge",
                    "parent_category": "Foundations of Wellbeing",
                    "positive_label": "melek huruf",
                    "negative_label": "buta huruf",
                    "description": "Atribut terkait akses terhadap pengetahuan dasar",
                },
                {
                    "pair_id": "sp_access_information",
                    "parent_category": "Foundations of Wellbeing",
                    "positive_label": "berpengetahuan luas",
                    "negative_label": "kurang informasi",
                    "description": "Atribut terkait akses dan ketersediaan informasi",
                },
                {
                    "pair_id": "sp_health_wellness",
                    "parent_category": "Foundations of Wellbeing",
                    "positive_label": "sehat",
                    "negative_label": "tidak sehat",
                    "description": "Atribut terkait status kesehatan dan kebugaran",
                },
                {
                    "pair_id": "sp_environment_quality",
                    "parent_category": "Foundations of Wellbeing",
                    "positive_label": "ramah lingkungan",
                    "negative_label": "perusak lingkungan",
                    "description": "Atribut terkait perilaku atau dampak terhadap lingkungan",
                },
                {
                    "pair_id": "sp_personal_freedom",
                    "parent_category": "Opportunity",
                    "positive_label": "menjunjung kebebasan",
                    "negative_label": "otoriter",
                    "description": "Atribut terkait kebebasan pribadi dalam bertindak dan berpikir",
                },
                {
                    "pair_id": "sp_freedom_choice",
                    "parent_category": "Opportunity",
                    "positive_label": "mandiri",
                    "negative_label": "konformis",
                    "description": "Atribut terkait pilihan hidup, kemandirian, atau tekanan sosial",
                },
                {
                    "pair_id": "sp_inclusiveness",
                    "parent_category": "Opportunity",
                    "positive_label": "inklusif",
                    "negative_label": "diskriminatif",
                    "description": "Atribut terkait keterbukaan dan penerimaan terhadap keberagaman",
                },
            ],
        },
    ],
}
