from faster_whisper import WhisperModel

from pathlib import Path
import os
import downloader

MODEL_LINKS = {
    "tiny": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/tiny-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/tiny-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/tiny-ct2-fp16.zip",
            ],
            "checksum": "3c7c0512b7b881ecb4cb0693d543aed2a9178968bef255fa0ca8b880541ec789"
        },
        #"bfloat16": {
        #    "urls": [
        #        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/tiny-ct2-bfp16.zip",
        #        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/tiny-ct2-bfp16.zip",
        #        "https://s3.libs.space:9000/ai-models/Whisper-CT2/tiny-ct2-bfp16.zip",
        #    ],
        #    "checksum": "a9967e0c2c2fae9385443d32db05f51e8695444d75ae0aee92b644adb57592f6"
        #},
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/tiny-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/tiny-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/tiny-ct2.zip",
            ],
            "checksum": "18f4d5a6dbb9d27b748ee7a58ef455ff6640f230e5d64781e9cfb16181136b04"
        }
    },
    "tiny.en": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/tiny.en-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/tiny.en-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/tiny.en-ct2-fp16.zip",
            ],
            "checksum": "a14fedc8e57090505ec46119d346895604f5a6b5a8a44a7a137c44169544ea99"
        },
        #"bfloat16": {
        #    "urls": [
        #        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/tiny.en-ct2-bfp16.zip",
        #        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/tiny.en-ct2-bfp16.zip",
        #        "https://s3.libs.space:9000/ai-models/Whisper-CT2/tiny.en-ct2-bfp16.zip",
        #    ],
        #    "checksum": "14362200375ed60a0f341f6076cd31302ac11752aedc5ca32d8d6b0bb6761f60"
        #},
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/tiny.en-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/tiny.en-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/tiny.en-ct2.zip",
            ],
            "checksum": "814c670c9922574c9e0e3be8d7f616e53347ec2dee099648523e2f88ec436eec"
        }
    },
    "base": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/base-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/base-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/base-ct2-fp16.zip",
            ],
            "checksum": "fa863d01b4ef07bab0467d13b33221c8e6273362078ec6268bbc6398f40c0ab4"
        },
        #"bfloat16": {
        #    "urls": [
        #        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/base-ct2-bfp16.zip",
        #        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/base-ct2-bfp16.zip",
        #        "https://s3.libs.space:9000/ai-models/Whisper-CT2/base-ct2-bfp16.zip",
        #    ],
        #    "checksum": "ffea322cf2d5a19b0fed1b14084f62fcaa20b9445b09dec0223ae6ddfcf386f6"
        #},
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/base-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/base-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/base-ct2.zip",
            ],
            "checksum": "e95001e10c40b57797e208f2e915e16d86bac67f204742bac2b8950e6eeb3539"
        }
    },
    "base.en": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/base.en-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/base.en-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/base.en-ct2-fp16.zip",
            ],
            "checksum": "ec00c31ef78f035950c276ff01e5da96b4e9761bc15e872b2ec02371ac357484"
        },
        #"bfloat16": {
        #    "urls": [
        #        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/base.en-ct2-bfp16.zip",
        #        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/base.en-ct2-bfp16.zip",
        #        "https://s3.libs.space:9000/ai-models/Whisper-CT2/base.en-ct2-bfp16.zip",
        #    ],
        #    "checksum": "61d9c0e1e03a0de5a676baebae5afd4bea8c31b58d8a8545eafe46fd39af4223"
        #},
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/base.en-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/base.en-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/base.en-ct2.zip",
            ],
            "checksum": "5113b44b8f4fe1927f935d85326df5bbe708ab269144fc9399234f9e9b9d61d1"
        }
    },
    "small": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/small-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/small-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/small-ct2-fp16.zip",
            ],
            "checksum": "9f0618523bf19dc68d99109ba319f2faba2c94ef9d063aa300115935f3d09f14"
        },
        #"bfloat16": {
        #    "urls": [
        #        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/small-ct2-bfp16.zip",
        #        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/small-ct2-bfp16.zip",
        #        "https://s3.libs.space:9000/ai-models/Whisper-CT2/small-ct2-bfp16.zip",
        #    ],
        #    "checksum": "e8144f9f11a661e0f4acf38b218007fe138b8600906275cfec25bb1a1fb2e8fe"
        #},
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/small-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/small-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/small-ct2.zip",
            ],
            "checksum": "b887054992cf42abddad057e4b52f3ef6b1a079485244d786f1941a6fec8c02e"
        }
    },
    "small.en": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/small.en-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/small.en-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/small.en-ct2-fp16.zip",
            ],
            "checksum": "9f0618523bf19dc68d99109ba319f2faba2c94ef9d063aa300115935f3d09f14"
        },
        #"bfloat16": {
        #    "urls": [
        #        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/small.en-ct2-bfp16.zip",
        #        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/small.en-ct2-bfp16.zip",
        #        "https://s3.libs.space:9000/ai-models/Whisper-CT2/small.en-ct2-bfp16.zip",
        #    ],
        #    "checksum": "5c761c734fa8a84823341fb6484dc04ae0e35a7cea8c8fd7749f248ac1a191ae"
        #},
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/small.en-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/small.en-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/small.en-ct2.zip",
            ],
            "checksum": "c7eeb56070467bfad17ec774f66ce8dfc0b601d9c2ad5f96b3e4da9331552692"
        }
    },
    "medium": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/medium-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/medium-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/medium-ct2-fp16.zip",
            ],
            "checksum": "13d2d91bdd2c3722c0592cbffca468992257eb3ddb782b1779c59091a4d91dd4"
        },
        #"bfloat16": {
        #    "urls": [
        #        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/medium-ct2-bfp16.zip",
        #        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/medium-ct2-bfp16.zip",
        #        "https://s3.libs.space:9000/ai-models/Whisper-CT2/medium-ct2-bfp16.zip",
        #    ],
        #    "checksum": "3b002336f5fcd025a0358c0dd57f96fd5ac8696951f4b17ed7744aec6cca8cf2"
        #},
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/medium-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/medium-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/medium-ct2.zip",
            ],
            "checksum": "5682a3833f4c87ed749778a844ccc9da6d8b3e3a2fef338cf5e66b495050e2e6"
        }
    },
    "medium.en": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/medium.en-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/medium.en-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/medium.en-ct2-fp16.zip",
            ],
            "checksum": "13d2d91bdd2c3722c0592cbffca468992257eb3ddb782b1779c59091a4d91dd4"
        },
        #"bfloat16": {
        #    "urls": [
        #        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/medium.en-ct2-bfp16.zip",
        #        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/medium.en-ct2-bfp16.zip",
        #        "https://s3.libs.space:9000/ai-models/Whisper-CT2/medium.en-ct2-bfp16.zip",
        #    ],
        #    "checksum": "07065633915b1d64e4e3d977043d4c61e20469ee26fe1e8fa2333611147c3605"
        #},
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/medium.en-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/medium.en-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/medium.en-ct2.zip",
            ],
            "checksum": "8bf93eb5018c44c9115b6b942f8bc518790f88c2db93920f2da1a6a1efefe002"
        }
    },
    "large-v1": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/large-v1-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/large-v1-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/large-v1-ct2-fp16.zip",
            ],
            "checksum": "42ecc70522602e69fe6365ef73173bbb1178ff8fd99210b96ea9025a205014bb"
        },
        #"bfloat16": {
        #    "urls": [
        #        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/large-v1-ct2-bfp16.zip",
        #        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/large-v1-ct2-bfp16.zip",
        #        "https://s3.libs.space:9000/ai-models/Whisper-CT2/large-v1-ct2-bfp16.zip",
        #    ],
        #    "checksum": "4d66ab08d2410711b82f529ab72d16a6b8806864d278cf20657e30ba59851667"
        #},
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/large-v1-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/large-v1-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/large-v1-ct2.zip",
            ],
            "checksum": "82bd59ee73d7b52f60de5566e8e3e429374bd2dd1bce3e2f6fc18b620dbcf0cf"
        }
    },
    "large-v2": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/large-v2-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/large-v2-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/large-v2-ct2-fp16.zip",
            ],
            "checksum": "2397ed6433a08d4b6968852bc1b761b488c3149a3a52f49b62b2ac60d1d5cef0"
        },
        #"bfloat16": {
        #    "urls": [
        #        "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/large-v2-ct2-bfp16.zip",
        #        "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/large-v2-ct2-bfp16.zip",
        #        "https://s3.libs.space:9000/ai-models/Whisper-CT2/large-v2-ct2-bfp16.zip",
        #    ],
        #    "checksum": "220190ce2b0b3a4512b742b8c439798ef6f6b04ac83b8c130da2f0a35569a2d7"
        #},
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/large-v2-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/large-v2-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/large-v2-ct2.zip",
            ],
            "checksum": "c9e889f59cacfef9ebe76a1db5d80befdcf0043195c07734f6984d19e78c8253"
        }
    },
    # Finetune Models
    "small.eu": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.eu-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.eu-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.eu-ct2-fp16.zip",
            ],
            "checksum": "1995e89cd91f8905cc58af91ffca79ca357c8ed2deb76a9b8b9b37b973bb6686"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.eu-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.eu-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.eu-ct2.zip",
            ],
            "checksum": "56d4a47df698ba532569efe62951ca78ef8dc9f8e9565fe0b1605eea1f7f3d0e"
        }
    },
    "medium.eu": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.eu-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.eu-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.eu-ct2-fp16.zip",
            ],
            "checksum": "8a0e05afcf804b5c5c2a0de2fe028073d9f26c84f7da774c18ba4baa12aafb6e"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.eu-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.eu-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.eu-ct2.zip",
            ],
            "checksum": "d56fd31e48e0be8cadc93a1afa8193d4663cd8b6c13dc14a919f79c9c122d721"
        }
    },
    "small.de": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.de-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.de-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.de-ct2-fp16.zip",
            ],
            "checksum": "95395a303666f61fa679a7565492b0b852b01773d9aec16fdcda2d523910f1a9"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.de-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.de-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.de-ct2.zip",
            ],
            "checksum": "fb182c96410c935b6b15d76aa36e809cb676a454c39b173bce8d9ffd0c9b9bc0"
        }
    },
    "medium.de": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.de-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.de-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.de-ct2-fp16.zip",
            ],
            "checksum": "304c23ff13e9674a450efc2e7f7dc4f5ee8ab879c813cc82f1a9cb65f94ec684"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.de-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.de-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.de-ct2.zip",
            ],
            "checksum": "f31cbd4771ad7dae5d9c7a3e6b516ca6053220fb0f35052dc0576733f8bec2a7"
        }
    },
    "large-v2.de2": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.de2-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.de2-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.de2-ct2-fp16.zip",
            ],
            "checksum": "1b95edeae4b38006f7b58ac3b356bc33a644ff2597f8a3502a0d3740937f2c2b"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.de2-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.de2-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.de2-ct2.zip",
            ],
            "checksum": "59f4f4e4c9ab05a56e7169636c47268f873189a35e3e313393dd81cc0d4477ce"
        }
    },
    "small.de-swiss": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.de-swiss-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.de-swiss-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.de-swiss-ct2-fp16.zip",
            ],
            "checksum": "3a78e7858798d2c544a78ca75886541d42d264497729605cbfad228b63bc0605"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.de-swiss-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.de-swiss-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.de-swiss-ct2.zip",
            ],
            "checksum": "43b80481a77893acee0d32275e388058569bbce8e2c610e502d32a8f1ebd57d4"
        }
    },
    "medium.mix-jpv2": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.mix-jpv2-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.mix-jpv2-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.mix-jpv2-ct2-fp16.zip",
            ],
            "checksum": "1ff8084947f2d0d549386763617c03e84ae7f0b5ce03e6ba609e2682842b4232"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.mix-jpv2-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.mix-jpv2-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.mix-jpv2-ct2.zip",
            ],
            "checksum": "184c6d9d54327fb272e74f1ef66a59b22e795141cd2918eff7422370171b7601"
        }
    },
    "large-v2.mix-jp": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.mix-jp-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.mix-jp-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.mix-jp-ct2-fp16.zip",
            ],
            "checksum": "774c884ace5c4e0d21ed6c49a552bffc7e7bc7ea1d14e7eb46778bf45a1b6228"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.mix-jp-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.mix-jp-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.mix-jp-ct2.zip",
            ],
            "checksum": "d91f73ba6a5eb7ab3c7527b51de51739476008befdd754fc8fc795065b580b12"
        }
    },
    "small.jp": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.jp-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.jp-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.jp-ct2-fp16.zip",
            ],
            "checksum": "014227a27aeceaaf3b3fe62e99ab9869d70013bb084e3050e66bea4c13f4d4b2"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.jp-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.jp-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.jp-ct2.zip",
            ],
            "checksum": "6817811e7549eca3637d11329c251e0e410f3ce373505bec0bc1a524a5157782"
        }
    },
    "medium.jp": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.jp-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.jp-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.jp-ct2-fp16.zip",
            ],
            "checksum": "888d57b356bf910e93c4cc40ea124357d087c7f191d99a27211728a118882994"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.jp-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.jp-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.jp-ct2.zip",
            ],
            "checksum": "ece7f37b0447aa08664df246ff6b449e6a77a27b645ce8ed22518bf5d12c1f10"
        }
    },
    "large-v2.jp": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.jp-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.jp-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.jp-ct2-fp16.zip",
            ],
            "checksum": "ae20c09d8698feb809a4f7e37aa1c3daea786d3bb0c572b5b861c722ef00082b"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.jp-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.jp-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.jp-ct2.zip",
            ],
            "checksum": "a555221eada1c35b0df8d4241c988509a525947a6fa3bdcaf8fa8ce8b833fc8c"
        }
    },
    "medium.ko": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.ko-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.ko-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.ko-ct2-fp16.zip",
            ],
            "checksum": "e1c8ee9478eff971b0360ec000b2571c63f0b90e48c13aa1d91d79c168807173"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.ko-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.ko-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.ko-ct2.zip",
            ],
            "checksum": "c93ac81a78a29da4acd10da6d7f5b6c92f0d495f12c5bbb60b7069fbed499834"
        }
    },
    "large-v2.ko": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.ko-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.ko-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.ko-ct2-fp16.zip",
            ],
            "checksum": "5c1638e38a3d8ffcbc2f09f92c381c7cca9a83fc5ad8f4e6fbe4891f10a91094"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.ko-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.ko-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.ko-ct2.zip",
            ],
            "checksum": "8878633856ac109e203e11265de60424e0b7a4471f20d54d5b2cc23efc896e01"
        }
    },
    "small.zh": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.zh-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.zh-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.zh-ct2-fp16.zip",
            ],
            "checksum": "80c62e46595ef69a57e35c67d7c80adda4fa7d5aa97bb8468c0a23d0d1877b8d"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.zh-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.zh-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.zh-ct2.zip",
            ],
            "checksum": "c349846850e13b68f64a851b3fda88113ad4e8f362dad92acd245fad434eece4"
        }
    },
    "medium.zh": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.zh-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.zh-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.zh-ct2-fp16.zip",
            ],
            "checksum": "3706cc6e5e2f6740cd34ababdb49191be69a3080b55e23e28d0c6bacae900fc8"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.zh-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.zh-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.zh-ct2.zip",
            ],
            "checksum": "69fd0221d27f0b84ecab019db65712766efa184f0f7c888c89193a1c153435de"
        }
    },
    "large-v2.zh": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.zh-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.zh-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.zh-ct2-fp16.zip",
            ],
            "checksum": "8ab152261bec1805c7420ba23cfab0467d86e02995eca9ac8cb08b393eeff90a"
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.zh-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.zh-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.zh-ct2.zip",
            ],
            "checksum": "d12692bb38167d534247cb9aca60a5766435292d5820049f4f257785b7c22a96"
        }
    },
}

TOKENIZER_LINKS = {
    "normal": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/tokenizer.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/tokenizer.zip",
            "https://s3.libs.space:9000/ai-models/Whisper-CT2/tokenizer.zip",
        ],
        "checksum": "f6233d181a04abce6e2ba20189d5872b58ce2e14917af525a99feb5619777d7d"
    },
    "en": {
        "urls": [
            "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/tokenizer.en.zip",
            "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/tokenizer.en.zip",
            "https://s3.libs.space:9000/ai-models/Whisper-CT2/tokenizer.en.zip",
        ],
        "checksum": "fb364e7cae84eedfd742ad116a397daa75e4eebba38f27e3f391ae4fee19afa9"
    }
}


def needs_download(model: str, compute_type: str = "float32"):
    model_cache_path = Path(".cache/whisper")
    model_path = Path(model_cache_path / (model + "-ct2"))
    if compute_type == "float16" or compute_type == "int8_float16" or compute_type == "int16" or compute_type == "int8":
        model_path = Path(model_cache_path / (model + "-ct2-fp16"))
    #if compute_type == "bfloat16" or compute_type == "int8_bfloat16":
    #    model_path = Path(model_cache_path / (model + "-ct2-bfp16"))

    pretrained_lang_model_file = Path(model_path / "model.bin")

    if not model_cache_path.exists() or not Path(model_path).exists() or not pretrained_lang_model_file.is_file():
        return True

    tokenizer_file = Path(model_path / "tokenizer.json")
    if not tokenizer_file.is_file():
        return True

    return False


def download_model(model: str, compute_type: str = "float32"):
    model_cache_path = Path(".cache/whisper")
    os.makedirs(model_cache_path, exist_ok=True)
    model_path = Path(model_cache_path / (model + "-ct2"))
    if compute_type == "float16" or compute_type == "int8_float16" or compute_type == "int16" or compute_type == "int8":
        compute_type = "float16"
        model_path = Path(model_cache_path / (model + "-ct2-fp16"))
    #if compute_type == "bfloat16" or compute_type == "int8_bfloat16":
    #    compute_type = "bfloat16"
    #    model_path = Path(model_cache_path / (model + "-ct2-bfp16"))


    pretrained_lang_model_file = Path(model_path / "model.bin")

    if not Path(model_path).exists() or not pretrained_lang_model_file.is_file():
        print("downloading faster-whisper...")
        if not downloader.download_extract(MODEL_LINKS[model][compute_type]["urls"],
                                           str(model_cache_path.resolve()),
                                           MODEL_LINKS[model][compute_type]["checksum"], title="Speech 2 Text (faster whisper)"):
            print("Model download failed")

    tokenizer_file = Path(model_path / "tokenizer.json")
    if not tokenizer_file.is_file() and Path(model_path).exists():
        tokenizer_type = "normal"
        if ".en" in model:
            tokenizer_type = "en"
        print("downloading tokenizer...")
        if not downloader.download_extract(TOKENIZER_LINKS[tokenizer_type]["urls"],
                                           str(model_path.resolve()),
                                           TOKENIZER_LINKS[tokenizer_type]["checksum"], title="tokenizer"):
            print("Tokenizer download failed")
    elif not Path(model_path).exists():
        print("no model downloaded for tokenizer.")


class FasterWhisper:
    model = None

    def __init__(self, model: str, device: str = "cpu", compute_type: str = "float32", cpu_threads: int = 0,
                 num_workers: int = 1):
        if self.model is None:
            self.load_model(model, device, compute_type, cpu_threads, num_workers)

    def load_model(self, model: str, device: str = "cpu", compute_type: str = "float32", cpu_threads: int = 0,
                   num_workers: int = 1):
        model_cache_path = Path(".cache/whisper")
        os.makedirs(model_cache_path, exist_ok=True)
        model_path = Path(model_cache_path / (model + "-ct2"))
        if compute_type == "float16" or compute_type == "int8_float16":
            model_path = Path(model_cache_path / (model + "-ct2-fp16"))
        #if compute_type == "bfloat16" or compute_type == "int8_bfloat16":
        #    model_path = Path(model_cache_path / (model + "-ct2-bfp16"))

        print(f"faster-whisper {model} is Loading to {device} using {compute_type} precision...")
        self.model = WhisperModel(str(Path(model_path).resolve()), device=device, compute_type=compute_type,
                                  cpu_threads=cpu_threads, num_workers=num_workers)

    def transcribe(self, audio_sample, task, language, condition_on_previous_text,
                   initial_prompt, logprob_threshold, no_speech_threshold,
                   temperature, beam_size, word_timestamps, without_timestamps, patience, length_penalty) -> dict:

        result_segments, audio_info = self.model.transcribe(audio_sample, task=task,
                                                            language=language,
                                                            condition_on_previous_text=condition_on_previous_text,
                                                            initial_prompt=initial_prompt,
                                                            log_prob_threshold=logprob_threshold,
                                                            no_speech_threshold=no_speech_threshold,
                                                            temperature=temperature,
                                                            beam_size=beam_size,
                                                            word_timestamps=word_timestamps,
                                                            without_timestamps=without_timestamps,
                                                            patience=patience,
                                                            length_penalty=length_penalty
                                                            )

        result = {
            'text': " ".join([segment.text for segment in result_segments]),
            'type': task,
            'language': audio_info.language
        }

        return result
