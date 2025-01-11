import gc

import torch
from faster_whisper import WhisperModel

from pathlib import Path
import os
import downloader
import settings
from Models.Singleton import SingletonMeta

MODEL_LINKS = {
    "tiny": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/tiny-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/tiny-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/tiny-ct2-fp16.zip",
            ],
            "checksum": "7256a358cbe500c912f503af2624e699a941cac7f1e4e21a5210aeec8c4b2fa7",
            "file_checksums": {
                "config.json": "e7347a2cdad4c72e93b3fc1ba6e105ef6b28f6b24d8d5caabe98827fed46472b",
                "model.bin": "816b390a889d6337c7a5d01861e80328dc9123a7a7c61fab1b991eb4abe7afd2",
                "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55",
                "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/tiny-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/tiny-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/tiny-ct2.zip",
            ],
            "checksum": "baf8710f3e254b4001a6b711518faba557ca829ceeb2941891f3a0bc450c1acd",
            "file_checksums": {
                "config.json": "e7347a2cdad4c72e93b3fc1ba6e105ef6b28f6b24d8d5caabe98827fed46472b",
                "model.bin": "cdeb97af4be8aeb09523936242406ab227f8891206567978d45027c841716d4c",
                "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55",
                "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"
            }
        }
    },
    "tiny.en": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/tiny.en-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/tiny.en-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/tiny.en-ct2-fp16.zip",
            ],
            "checksum": "061cdd9438c7b85d8f33fc836b881320024934d509ada679e39dd5b7ff93efcf",
            "file_checksums": {
                "config.json": "1323b2d5bbe66bb6dc08298a111acd094484d83a6ffd973811b80720191a6277",
                "model.bin": "37ce70b969d78e8fa7c36e1018e551aac29b5118c754e009eac1481b9fcb64de",
                "tokenizer.json": "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df",
                "vocabulary.json": "561cc18c4b9a8b41a2ecfc3b5092b62a24d1bb6e79998ca1bb45326353647874",
                "vocabulary.txt": "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/tiny.en-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/tiny.en-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/tiny.en-ct2.zip",
            ],
            "checksum": "cc15d1122fa629841b01a04c822316748ed43ca6f73b4185ee72dbff7c3fae40",
            "file_checksums": {
                "config.json": "1323b2d5bbe66bb6dc08298a111acd094484d83a6ffd973811b80720191a6277",
                "model.bin": "10dd0e24e607d467631e51d3ce2a1e5ddf915d1caff909afaf9466211aee41b4",
                "tokenizer.json": "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df",
                "vocabulary.json": "561cc18c4b9a8b41a2ecfc3b5092b62a24d1bb6e79998ca1bb45326353647874",
                "vocabulary.txt": "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf"
            }
        }
    },
    "base": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/base-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/base-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/base-ct2-fp16.zip",
            ],
            "checksum": "ec16d2cc98a25bce05af9473d6cb3fd4936c0eacc685748bb246f1ffc56d934a",
            "file_checksums": {
                "config.json": "d01989ec688fe3508788cdee0dd4f6b2de5ce354b2c0b5f585d713fc19fab81f",
                "model.bin": "b22573c18c91a4f92340ad824031549a9d5d86ba786e4fc7c8844988013f2171",
                "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55",
                "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/base-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/base-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/base-ct2.zip",
            ],
            "checksum": "eaab0e214be64d3a473a859514f4748e2359e91245b908a6125f4b10a7f22666",
            "file_checksums": {
                "config.json": "d01989ec688fe3508788cdee0dd4f6b2de5ce354b2c0b5f585d713fc19fab81f",
                "model.bin": "d71c1cf38f3322b5137c65de9c4b63c7fd0a18d885942c9373af8afc9057816c",
                "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55",
                "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"
            }
        }
    },
    "base.en": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/base.en-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/base.en-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/base.en-ct2-fp16.zip",
            ],
            "checksum": "d90a85ac48fb27ffb5e2ece43749667e17e03b18ab6969747d0aba44d138d26f",
            "file_checksums": {
                "config.json": "4c4ab959237ad00b2b76eae0fc2a62b2232720022fafd98366a53e577056d132",
                "model.bin": "25edce04a2ae0d8a43b1adcdaafc8c0c8c2592ae61820b01bf533e5832591094",
                "tokenizer.json": "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df",
                "vocabulary.json": "561cc18c4b9a8b41a2ecfc3b5092b62a24d1bb6e79998ca1bb45326353647874",
                "vocabulary.txt": "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/base.en-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/base.en-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/base.en-ct2.zip",
            ],
            "checksum": "64346451c108e8086553d92e1f14f2b8333c97f9d2e32c80976bf188cddc05ab",
            "file_checksums": {
                "config.json": "4c4ab959237ad00b2b76eae0fc2a62b2232720022fafd98366a53e577056d132",
                "model.bin": "b6d10875fdb995f639fa968ce6cae91d677f20415e3c1aa058bcc86ba353d356",
                "tokenizer.json": "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df",
                "vocabulary.json": "561cc18c4b9a8b41a2ecfc3b5092b62a24d1bb6e79998ca1bb45326353647874",
                "vocabulary.txt": "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf"
            }
        }
    },
    "small": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/small-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/small-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/small-ct2-fp16.zip",
            ],
            "checksum": "58021a0635469d240b9afce7360a191f401bf6151bde527301500083a007ebe3",
            "file_checksums": {
                "config.json": "364e66f09ce360358f61e549786419eedc8e6ab709186304dc77f84a3c4966c8",
                "model.bin": "d9852d4a80538ea41815e1d3c3e47adece75bb946416eaefbc45771bf935efbc",
                "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55",
                "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/small-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/small-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/small-ct2.zip",
            ],
            "checksum": "9b5da5321dc44750eb7ecc84e77278f9e35b7920ab70af60ddbdf4760f10b840",
            "file_checksums": {
                "config.json": "364e66f09ce360358f61e549786419eedc8e6ab709186304dc77f84a3c4966c8",
                "model.bin": "51ed2f610a8dbf2da7510ff2e753be8c132e02f2a53791536a813c27c11ab3b2",
                "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55",
                "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"
            }
        }
    },
    "small.en": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/small.en-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/small.en-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/small.en-ct2-fp16.zip",
            ],
            "checksum": "e269f5524a8d4a127832e4c224be3bfd97a03fb339cefe70dc537fcce72c8dbd",
            "file_checksums": {
                "vocabulary.json": "561cc18c4b9a8b41a2ecfc3b5092b62a24d1bb6e79998ca1bb45326353647874",
                "tokenizer.json": "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df",
                "config.json": "19104aa7792d33bca76f5ae1008dbfb889c27c9c32fe46babea7747bafb479d6",
                "model.bin": "207faf224115546fbc7d5f6c9752a492b9dc3e0ac24cb91b1a6a054d376e908a",
                "vocabulary.txt": "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf",
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/small.en-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/small.en-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/small.en-ct2.zip",
            ],
            "checksum": "9ad75ef05dd37fa05f1c90297418af29a2bfc4fcd409aca1137f6766f82c7e2e",
            "file_checksums": {
                "vocabulary.json": "561cc18c4b9a8b41a2ecfc3b5092b62a24d1bb6e79998ca1bb45326353647874",
                "tokenizer.json": "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df",
                "config.json": "19104aa7792d33bca76f5ae1008dbfb889c27c9c32fe46babea7747bafb479d6",
                "model.bin": "bb5166a3aa481847c008f77361aa038f23ff4450e766f7bfa7fa85fae9b9e4a4",
                "vocabulary.txt": "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf",
            }
        }
    },
    "medium": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/medium-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/medium-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/medium-ct2-fp16.zip",
            ],
            "checksum": "c8b8d4e9c936cf67041fa6ef7b910fec821fc69e01fc63ddaf953c2ae5dad588",
            "file_checksums": {
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55",
                "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
                "config.json": "8c2bab8a9c4491a048e03899fb01efcca85bef1a971eaacc3921b3a0348b10c3",
                "model.bin": "e4bc4b8bfeda89e03164a815b29662396da0c2f630e320e407696607b6802744",
                "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/medium-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/medium-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/medium-ct2.zip",
            ],
            "checksum": "102c89e75dfd4186476cf6c3881541302b41644083adef2e6a05570d2168d83c",
            "file_checksums": {
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55",
                "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
                "config.json": "8c2bab8a9c4491a048e03899fb01efcca85bef1a971eaacc3921b3a0348b10c3",
                "model.bin": "5a403b900fde4054bde3067e4962396a1e3c1b80308a33121b04a16c6a3bf70a",
                "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
            }
        }
    },
    "medium.en": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/medium.en-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/medium.en-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/medium.en-ct2-fp16.zip",
            ],
            "checksum": "558c09344e8dd0e2e065cbbe53c71ba88691402fe60153619dd4db3f0482c7ec",
            "file_checksums": {
                "vocabulary.json": "561cc18c4b9a8b41a2ecfc3b5092b62a24d1bb6e79998ca1bb45326353647874",
                "tokenizer.json": "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df",
                "config.json": "0f5732de422b1c161c447cb0ef3b48aa712a5c6c71025073a2f4283e10502452",
                "model.bin": "73c1bbea6b51e705d2375f75ecf83c6e708717ecc0d392744d4762892c642e13",
                "vocabulary.txt": "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf",
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/medium.en-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/medium.en-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/medium.en-ct2.zip",
            ],
            "checksum": "f9ba00c694c0e1bd9fba736f6febaaa41aedb5ebe55580ed412697358e75338b",
            "file_checksums": {
                "vocabulary.json": "561cc18c4b9a8b41a2ecfc3b5092b62a24d1bb6e79998ca1bb45326353647874",
                "tokenizer.json": "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df",
                "config.json": "0f5732de422b1c161c447cb0ef3b48aa712a5c6c71025073a2f4283e10502452",
                "model.bin": "6dbcfb29f25ee5f01e845ec58d0015638b4dfcc42c799896bea6b82733e0eeb7",
                "vocabulary.txt": "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf",
            }
        }
    },
    "large-v1": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/large-v1-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/large-v1-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/large-v1-ct2-fp16.zip",
            ],
            "checksum": "d3082bdb26acdcd248fb718ebc0fda72299bf6cc2da3487cf4232019c0e819d0",
            "file_checksums": {
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55",
                "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
                "config.json": "a8022ab5b17a7ae6187b9d9405ff67119b4f2be0d0313c329dfb486f00b622aa",
                "model.bin": "7a31bc6973611750511d2dfc867903f21bcb8d686f8dbaab47b541099c4f05ee",
                "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/large-v1-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/large-v1-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/large-v1-ct2.zip",
            ],
            "checksum": "189cc6cc6c54986c699b67700718029fede767d3731a67c0939f3a323615aa5b",
            "file_checksums": {
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55",
                "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
                "config.json": "a8022ab5b17a7ae6187b9d9405ff67119b4f2be0d0313c329dfb486f00b622aa",
                "model.bin": "0a5dc8ea4e48a636651a85feb85ac01cb3077d9b9707dd5c9d5482fef93da45b",
                "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
            }
        }
    },
    "large-v2": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/large-v2-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/large-v2-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/large-v2-ct2-fp16.zip",
            ],
            "checksum": "f554f5a9c992348968209cfbf2b3037fc64bdbe18fc060fe914044e2d2106324",
            "file_checksums": {
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55",
                "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
                "config.json": "6a8982762acd041d98336854dda81afe09b8ea2472d41c4ec99a3651547382e6",
                "model.bin": "b2c0dfbe0c154897f26c0b0b73141fd159b935d8010295e5191ea7dbe085c6bf",
                "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/new1/large-v2-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/new1/large-v2-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/new1/large-v2-ct2.zip",
            ],
            "checksum": "82f272ac2efa1ecceea8bcc025dc32f2a5e4087d184aadefb750826486061ae9",
            "file_checksums": {
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55",
                "tokenizer.json": "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab",
                "config.json": "6a8982762acd041d98336854dda81afe09b8ea2472d41c4ec99a3651547382e6",
                "model.bin": "1b0897377bdb8b0955a25ae3bf5e3e4494958ace10fec1568ca6c81eaa479989",
                "vocabulary.txt": "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913",
            }
        }
    },
    # float16 vs float32 converted model showed no difference...
    "large-v3": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/large-v3-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/large-v3-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/large-v3-ct2-fp16.zip",
            ],
            "checksum": "0521e0ee741b114674b146d048251520a51b0342da5de2bfd76e2470c18b84b7",
            "file_checksums": {
                "generation_config.json": "1149807b43a0dd788e052bfcb47c012b0b182946b66c63b3ecdf9aad2d9b5f66",
                "config.json": "b5b4368433a25df0943929beaf6833db03b767b150990ee078fe62c5a7b31434",
                "preprocessor_config.json": "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711",
                "tokenizer.json": "6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca",
                "model.bin": "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1",
                "vocabulary.json": "697ee5a65726e46b4d294d1a243d98e1878f5d81caba17891b1c5ebab7a912a9",
            }
        }
    },
    "large-v3-turbo": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/large-v3-turbo-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/large-v3-turbo-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/large-v3-turbo-ct2-fp16.zip",
            ],
            "checksum": "d16072c18d63c7b070d9f44f1a5c0d6a6fd36f13faa0c3bf41ce34fdc90cf9b8",
            "file_checksums": {
                "config.json": "a0feddc18de0c285ed147e5483c8d1bc911bd45e23104ae0726d79594e7a6b1d",
                "model.bin": "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da",
                "preprocessor_config.json": "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711",
                "tokenizer.json": "b3c8202bbf06d8ee4232c5984baa563784ac4737e2e7fdc42fa180200d3cfcdb",
                "vocabulary.json": "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1",
            }
        }
    },
    "medium-distilled.en": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/medium-distilled.en-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/medium-distilled.en-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/medium-distilled.en-ct2-fp16.zip",
            ],
            "checksum": "237de540a5a606dae47c61231b489ad3e43ab0750ce58f7921f0a0fadf4cf9d0",
            "file_checksums": {
                "config.json": "8fbd09f4b35e38db19b1cf43ed4d812b8da7c6b791aafbc93b37dc82f198ab70",
                "model.bin": "d4cb75d823dcd2647191064da76f026774c06c036908f38456165368d0e2d66a",
                "tokenizer.json": "79eb6cded9fd5d31cc83def0bc5fc9b3afa4973df1da4c7fbc89c44adfa8fe32",
                "vocabulary.json": "4dadfee7c4a871665f65c06037f5f5ec893fb2d7f5eb4cf11063618e31dbe11a"
            }
        },
    },
    "large-distilled-v2.en": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/large-distilled-v2.en-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/large-distilled-v2.en-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/large-distilled-v2.en-ct2-fp16.zip",
            ],
            "checksum": "3e0bcbc905259a61db35afa35d4559ba5284320cdcb44b9e7f0ebfc6701fed1d",
            "file_checksums": {
                "config.json": "721df87877126490ba0060bbecac3e75d66ea49daceed46c376c59680cdd922f",
                "model.bin": "b415593cd109c0418d42ecb106c40ae45c8229cb4090e67db12243901e071ab3",
                "tokenizer.json": "56633aee7338579d67d4ee85651f79f690c640d13a652465e5afbe79c437dfa8",
                "vocabulary.json": "5ad28279db5e546349708b9a74736e2c018737ac5a600f1048f1f26c99b85b47"
            }
        },
    },
    "large-distilled-v3.en": {
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/large-distilled-v3.en-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/large-distilled-v3.en-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/large-distilled-v3.en-ct2.zip",
            ],
            "checksum": "a1b202a5a281a5d488768b58961b5d9cfe4505136623f4aefb03478e43918e52",
            "file_checksums": {
                "added_tokens.json": "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0",
                "config.json": "90c55f775cc4e0bb17293d0bf12f96557a486f20dea886fabd8e6075a3588b21",
                "merges.txt": "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6",
                "model.bin": "8ea78a126adc145e020901f8891ad3536684e2754fcae77fc241f628298fd785",
                "normalizer.json": "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd",
                "preprocessor_config.json": "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711",
                "special_tokens_map.json": "baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74",
                "tokenizer_config.json": "844b642c73a91359722f47b35705f7174686df33d252695d8572cf9ac03a6389",
                "vocab.json": "e2aa043ef015641d363d8288e7c241c85e36a5c761fb303598e0710233344387",
                "vocabulary.json": "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1"
            }
        },
    },
    # Crisper Whisper https://github.com/nyrahealth/CrisperWhisper
    "crisper": {
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/crisper-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/crisper-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/crisper-ct2.zip",
            ],
            "checksum": "b6fa1d80e0ca1d6879dc5eda713892e177b1f89c2585b63b4b04e1c82b01d746",
            "file_checksums": {
                "config.json": "a7e36fb7fdf5c1f773903bedfe09f0eb330a7668cec92a2fa6369b40c25af847",
                "model.bin": "151eaa61fad9906491d6a7e0c2f152907276e5e318526ef7ffe6b55120213628",
                "preprocessor_config.json": "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711",
                "tokenizer.json": "86826f8033cf57f2b9f5643191b0aba1bba4cdc8307d2c5199e7e3d889bd6c9e",
                "vocabulary.json": "6286f6c910c88609702f41ab640dc8e274319d6868992de33d7957d44e95ef0a"
            }
        },
    },
    # Finetune Models
    "small.eu": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.eu-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.eu-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.eu-ct2-fp16.zip",
            ],
            "checksum": "1995e89cd91f8905cc58af91ffca79ca357c8ed2deb76a9b8b9b37b973bb6686",
            "file_checksums": {
                "config.json": "a37cee2f38f7621011b7225df53d2be862cb92204cd0c832777f9124d0e1d698",
                "model.bin": "e599a40fc5efc99d63708953979927f272f6a98a789a985b46e63c171288c148",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.eu-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.eu-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.eu-ct2.zip",
            ],
            "checksum": "56d4a47df698ba532569efe62951ca78ef8dc9f8e9565fe0b1605eea1f7f3d0e",
            "file_checksums": {
                "config.json": "a37cee2f38f7621011b7225df53d2be862cb92204cd0c832777f9124d0e1d698",
                "model.bin": "863afaa03b26ba494668bcf7e45272e3fb08fec414f32fd4282079151620253f",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55"
            }
        }
    },
    "medium.eu": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.eu-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.eu-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.eu-ct2-fp16.zip",
            ],
            "checksum": "8a0e05afcf804b5c5c2a0de2fe028073d9f26c84f7da774c18ba4baa12aafb6e",
            "file_checksums": {
                "config.json": "66405922f47151019e6444d77c63baf22f51c6f87f8ade31f6bd265676d7c4fc",
                "model.bin": "84d1bbce46f3bd9df4bbcc0ad165f3e9c5406f441cd74b9ca6f0e6f4735a0408",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.eu-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.eu-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.eu-ct2.zip",
            ],
            "checksum": "d56fd31e48e0be8cadc93a1afa8193d4663cd8b6c13dc14a919f79c9c122d721",
            "file_checksums": {
                "config.json": "66405922f47151019e6444d77c63baf22f51c6f87f8ade31f6bd265676d7c4fc",
                "model.bin": "a02a3d56486a50a7a659bad8bd10a825683d95a7407ecd72f10f414a1fe4898d",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55"
            }
        }
    },
    "small.de": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.de-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.de-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.de-ct2-fp16.zip",
            ],
            "checksum": "95395a303666f61fa679a7565492b0b852b01773d9aec16fdcda2d523910f1a9",
            "file_checksums": {
                "config.json": "a37cee2f38f7621011b7225df53d2be862cb92204cd0c832777f9124d0e1d698",
                "model.bin": "8d1ec85dbad2554a5eba334ee3f1234d448fca5f8df04e7e5836d92404d5037d",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.de-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.de-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.de-ct2.zip",
            ],
            "checksum": "fb182c96410c935b6b15d76aa36e809cb676a454c39b173bce8d9ffd0c9b9bc0",
            "file_checksums": {
                "config.json": "a37cee2f38f7621011b7225df53d2be862cb92204cd0c832777f9124d0e1d698",
                "model.bin": "f91425ad22f7a7c2236e15f2cf25f1b0c8187aa003538e0f1a81a24d8579f04c",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        }
    },
    "medium.de": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.de-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.de-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.de-ct2-fp16.zip",
            ],
            "checksum": "304c23ff13e9674a450efc2e7f7dc4f5ee8ab879c813cc82f1a9cb65f94ec684",
            "file_checksums": {
                "config.json": "66405922f47151019e6444d77c63baf22f51c6f87f8ade31f6bd265676d7c4fc",
                "model.bin": "3616e5d83cc7166edf487422aaa8a68aaed785f905d0551f30652ecc0a75c768",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.de-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.de-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.de-ct2.zip",
            ],
            "checksum": "f31cbd4771ad7dae5d9c7a3e6b516ca6053220fb0f35052dc0576733f8bec2a7",
            "file_checksums": {
                "config.json": "66405922f47151019e6444d77c63baf22f51c6f87f8ade31f6bd265676d7c4fc",
                "model.bin": "4ebc96b53cedabcd6546299a651f11b60582c9eb126156dc8a38e8d3afac3f57",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        }
    },
    "large-v2.de2": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.de2-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.de2-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.de2-ct2-fp16.zip",
            ],
            "checksum": "1b95edeae4b38006f7b58ac3b356bc33a644ff2597f8a3502a0d3740937f2c2b",
            "file_checksums": {
                "config.json": "5a6d68cc5d1ccd2fbc76c8930c73ca65092e2b9cdd28fa7db0a17cdeda1c4daa",
                "model.bin": "84fb70dc0ac967967136db4c37f0192512c5933000747caf8fff2bfebf6a78f8",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.de2-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.de2-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.de2-ct2.zip",
            ],
            "checksum": "59f4f4e4c9ab05a56e7169636c47268f873189a35e3e313393dd81cc0d4477ce",
            "file_checksums": {
                "config.json": "5a6d68cc5d1ccd2fbc76c8930c73ca65092e2b9cdd28fa7db0a17cdeda1c4daa",
                "model.bin": "44e21fbe384b25f25b42775796d07524d97af75391ae537ff0e8cde4f38e0635",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55"
            }
        }
    },
    "large-distilled-v3.de": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/large-distilled-v3.de-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/large-distilled-v3.de-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/large-distilled-v3.de-ct2-fp16.zip",
            ],
            "checksum": "20c992544b3842c6e349945f8adbf7ddea6e90d711fc5b0e28b26223303295db",
            "file_checksums": {
                "config.json": "f277d9390e65adc1630b349151eca38aaf7420c41c814271928854edcfd6ffbe",
                "model.bin": "03cd956827fa3424d8c9891329c92f2042cb7b27ca33b1742e4ec181064c4c4a",
                "vocabulary.json": "697ee5a65726e46b4d294d1a243d98e1878f5d81caba17891b1c5ebab7a912a9"
            }
        },
    },
    "small.de-swiss": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.de-swiss-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.de-swiss-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.de-swiss-ct2-fp16.zip",
            ],
            "checksum": "3a78e7858798d2c544a78ca75886541d42d264497729605cbfad228b63bc0605",
            "file_checksums": {
                "config.json": "c2e1a9518b8024e9e58c0bea7e99d1253938c89095de35730c64c44ea0c88cf9",
                "model.bin": "fae6f1b655a4687b2e8e6e006067e27f5b44031d53be0b3a84f97dfc88f907e9",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.de-swiss-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.de-swiss-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.de-swiss-ct2.zip",
            ],
            "checksum": "43b80481a77893acee0d32275e388058569bbce8e2c610e502d32a8f1ebd57d4",
            "file_checksums": {
                "config.json": "c2e1a9518b8024e9e58c0bea7e99d1253938c89095de35730c64c44ea0c88cf9",
                "model.bin": "4d13f18612a7a106b350afad1e446052039857fa24c56d5c88ffa21549ccdf6d",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55"
            }
        }
    },
    "medium.mix-jpv2": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.mix-jpv2-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.mix-jpv2-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.mix-jpv2-ct2-fp16.zip",
            ],
            "checksum": "1ff8084947f2d0d549386763617c03e84ae7f0b5ce03e6ba609e2682842b4232",
            "file_checksums": {
                "config.json": "7b2c8d25016ff196d8f9969078eaf4bf3fe2be63d14f5d26c6d269442110d231",
                "model.bin": "0d1b193cdd7e930a6307a0273c6e0f02aeffa2a0b597c8abb1d083a05ecfab4a",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.mix-jpv2-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.mix-jpv2-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.mix-jpv2-ct2.zip",
            ],
            "checksum": "184c6d9d54327fb272e74f1ef66a59b22e795141cd2918eff7422370171b7601",
            "file_checksums": {
                "config.json": "7b2c8d25016ff196d8f9969078eaf4bf3fe2be63d14f5d26c6d269442110d231",
                "model.bin": "cf2c004b538849e716e4e87d5bb245278da91317907ca43bd7c2328a0181efcf",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        }
    },
    "large-v2.mix-jp": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.mix-jp-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.mix-jp-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.mix-jp-ct2-fp16.zip",
            ],
            "checksum": "774c884ace5c4e0d21ed6c49a552bffc7e7bc7ea1d14e7eb46778bf45a1b6228",
            "file_checksums": {
                "config.json": "3377ad3386248c6faca4f76f5bbe8430040ec4a98e9121362767b9d1d4072f71",
                "model.bin": "167b914971d787440121ed5fa865fb28f39bb141d68276c7fc1e89fa8d8901d9",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.mix-jp-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.mix-jp-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.mix-jp-ct2.zip",
            ],
            "checksum": "d91f73ba6a5eb7ab3c7527b51de51739476008befdd754fc8fc795065b580b12",
            "file_checksums": {
                "config.json": "3377ad3386248c6faca4f76f5bbe8430040ec4a98e9121362767b9d1d4072f71",
                "model.bin": "bef33b4e0cd2c0438e02e0d358c5d042378a0d34dcfcd3ff2a3a5b440ac4356d",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        }
    },
    "small.jp": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.jp-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.jp-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.jp-ct2-fp16.zip",
            ],
            "checksum": "014227a27aeceaaf3b3fe62e99ab9869d70013bb084e3050e66bea4c13f4d4b2",
            "file_checksums": {
                "config.json": "c2e1a9518b8024e9e58c0bea7e99d1253938c89095de35730c64c44ea0c88cf9",
                "model.bin": "a30b6f0ac28b4c84d8b364c654d9249ff62e0e4c58d44a9d2a1d97285f76311d",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.jp-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.jp-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.jp-ct2.zip",
            ],
            "checksum": "6817811e7549eca3637d11329c251e0e410f3ce373505bec0bc1a524a5157782",
            "file_checksums": {
                "config.json": "c2e1a9518b8024e9e58c0bea7e99d1253938c89095de35730c64c44ea0c88cf9",
                "model.bin": "e99ed68e7f7d9ebb9d6f8234055e0c237a4788582e3ff35bb3e78c0f197ee37d",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        }
    },
    "medium.jp": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.jp-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.jp-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.jp-ct2-fp16.zip",
            ],
            "checksum": "888d57b356bf910e93c4cc40ea124357d087c7f191d99a27211728a118882994",
            "file_checksums": {
                "config.json": "7b2c8d25016ff196d8f9969078eaf4bf3fe2be63d14f5d26c6d269442110d231",
                "model.bin": "bebef5516a6ba1219d1ca95afe125ce190e4fc45c1dbdc533df174bc3484ff94",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.jp-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.jp-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.jp-ct2.zip",
            ],
            "checksum": "ece7f37b0447aa08664df246ff6b449e6a77a27b645ce8ed22518bf5d12c1f10",
            "file_checksums": {
                "config.json": "7b2c8d25016ff196d8f9969078eaf4bf3fe2be63d14f5d26c6d269442110d231",
                "model.bin": "ab15f6fdf6982dd29fab7bd75e5d01264b3bac56339317a04ff8ac5fb43c111a",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        }
    },
    "large-v2.jp": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.jp-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.jp-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.jp-ct2-fp16.zip",
            ],
            "checksum": "ae20c09d8698feb809a4f7e37aa1c3daea786d3bb0c572b5b861c722ef00082b",
            "file_checksums": {
                "config.json": "3377ad3386248c6faca4f76f5bbe8430040ec4a98e9121362767b9d1d4072f71",
                "model.bin": "981733951d560c82b0c4b9b305eabc7e88b4ed821758692d19581ed2239143ac",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.jp-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.jp-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.jp-ct2.zip",
            ],
            "checksum": "a555221eada1c35b0df8d4241c988509a525947a6fa3bdcaf8fa8ce8b833fc8c",
            "file_checksums": {
                "config.json": "3377ad3386248c6faca4f76f5bbe8430040ec4a98e9121362767b9d1d4072f71",
                "model.bin": "e4dfed5248054ae8a1c0e7c2e6abd04140046c9c6a536fa47821b0822843e753",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        }
    },
    "medium.ko": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.ko-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.ko-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.ko-ct2-fp16.zip",
            ],
            "checksum": "e1c8ee9478eff971b0360ec000b2571c63f0b90e48c13aa1d91d79c168807173",
            "file_checksums": {
                "config.json": "7b2c8d25016ff196d8f9969078eaf4bf3fe2be63d14f5d26c6d269442110d231",
                "model.bin": "136552263605f8fcb10160f0fc77a96c5208d0ab5d8198b9fbcd273e6abcb13d",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.ko-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.ko-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.ko-ct2.zip",
            ],
            "checksum": "c93ac81a78a29da4acd10da6d7f5b6c92f0d495f12c5bbb60b7069fbed499834",
            "file_checksums": {
                "config.json": "7b2c8d25016ff196d8f9969078eaf4bf3fe2be63d14f5d26c6d269442110d231",
                "model.bin": "60c1cecb9c43b07e31d202b216c5d9b283d4066d3092e0f24794cf67d9fce073",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        }
    },
    "large-v2.ko": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.ko-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.ko-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.ko-ct2-fp16.zip",
            ],
            "checksum": "5c1638e38a3d8ffcbc2f09f92c381c7cca9a83fc5ad8f4e6fbe4891f10a91094",
            "file_checksums": {
                "config.json": "23bc5a74663967d22e0ba2ef68bd781406eddfed1809f1b6d9bd72aaad9f2aec",
                "model.bin": "c143ba03aa4c4807a89bb5acc4cd3f967427eb9fc076481759849282b71ae915",
                "tokenizer.json": "ae5da9b8ecc0005c56bbb677bf48d93e4216a0dbcba7d34de73e37903cead706",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.ko-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.ko-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.ko-ct2.zip",
            ],
            "checksum": "8878633856ac109e203e11265de60424e0b7a4471f20d54d5b2cc23efc896e01",
            "file_checksums": {
                "config.json": "23bc5a74663967d22e0ba2ef68bd781406eddfed1809f1b6d9bd72aaad9f2aec",
                "model.bin": "c90a3b1b4402d7b525a29cf7021785bf66eb14b3bc9292327096b4298c8baf68",
                "tokenizer.json": "ae5da9b8ecc0005c56bbb677bf48d93e4216a0dbcba7d34de73e37903cead706",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55"
            }
        }
    },
    "small.zh": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.zh-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.zh-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.zh-ct2-fp16.zip",
            ],
            "checksum": "80c62e46595ef69a57e35c67d7c80adda4fa7d5aa97bb8468c0a23d0d1877b8d",
            "file_checksums": {
                "config.json": "a37cee2f38f7621011b7225df53d2be862cb92204cd0c832777f9124d0e1d698",
                "model.bin": "be78b583f9a2cd4c613851835c30aafd2a4dbb6014d95348cad4a2078315c714",
                "tokenizer.json": "c6dd51646033d560bae6b99bde79c6990344ebac51533d392af2c97b5ec8e632",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/small.zh-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/small.zh-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/small.zh-ct2.zip",
            ],
            "checksum": "c349846850e13b68f64a851b3fda88113ad4e8f362dad92acd245fad434eece4",
            "file_checksums": {
                "config.json": "a37cee2f38f7621011b7225df53d2be862cb92204cd0c832777f9124d0e1d698",
                "model.bin": "626ca2a558aec218b1e988e2eb985e1b8ee830c798b2fa0fdf57c3b2a8679816",
                "tokenizer.json": "c6dd51646033d560bae6b99bde79c6990344ebac51533d392af2c97b5ec8e632",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55"
            }
        }
    },
    "medium.zh": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.zh-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.zh-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.zh-ct2-fp16.zip",
            ],
            "checksum": "3706cc6e5e2f6740cd34ababdb49191be69a3080b55e23e28d0c6bacae900fc8",
            "file_checksums": {
                "config.json": "66405922f47151019e6444d77c63baf22f51c6f87f8ade31f6bd265676d7c4fc",
                "model.bin": "ac63f96a0c64a6b22cbf6c809b156817373dbb6620181e80917576acac26a9e7",
                "tokenizer.json": "c6dd51646033d560bae6b99bde79c6990344ebac51533d392af2c97b5ec8e632",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/medium.zh-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/medium.zh-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/medium.zh-ct2.zip",
            ],
            "checksum": "69fd0221d27f0b84ecab019db65712766efa184f0f7c888c89193a1c153435de",
            "file_checksums": {
                "config.json": "66405922f47151019e6444d77c63baf22f51c6f87f8ade31f6bd265676d7c4fc",
                "model.bin": "6f2909d0fae5a0833d668f787eb10365444dc410acaeeceee9fad2547909c90a",
                "tokenizer.json": "c6dd51646033d560bae6b99bde79c6990344ebac51533d392af2c97b5ec8e632",
                "vocabulary.json": "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55"
            }
        }
    },
    "large-v2.zh": {
        "float16": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.zh-ct2-fp16.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.zh-ct2-fp16.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.zh-ct2-fp16.zip",
            ],
            "checksum": "8ab152261bec1805c7420ba23cfab0467d86e02995eca9ac8cb08b393eeff90a",
            "file_checksums": {
                "config.json": "23bc5a74663967d22e0ba2ef68bd781406eddfed1809f1b6d9bd72aaad9f2aec",
                "model.bin": "a70bfd3c421e00b05861b25eaff7e772c6d152d0bc54ee7ebe8aeb61e55b28a9",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
        },
        "float32": {
            "urls": [
                "https://usc1.contabostorage.com/8fcf133c506f4e688c7ab9ad537b5c18:ai-models/Whisper-CT2/finetune/large-v2.zh-ct2.zip",
                "https://eu2.contabostorage.com/bf1a89517e2643359087e5d8219c0c67:ai-models/Whisper-CT2/finetune/large-v2.zh-ct2.zip",
                "https://s3.libs.space:9000/ai-models/Whisper-CT2/finetune/large-v2.zh-ct2.zip",
            ],
            "checksum": "d12692bb38167d534247cb9aca60a5766435292d5820049f4f257785b7c22a96",
            "file_checksums": {
                "config.json": "23bc5a74663967d22e0ba2ef68bd781406eddfed1809f1b6d9bd72aaad9f2aec",
                "model.bin": "1035785a7d3039a16f473e01763da5aaa00ab4038b19e714440435ac2d8cd084",
                "vocabulary.json": "ec4bc27240fb6d7bb8b3d084a1257d98c0b04c16f76142ca0263459ce04436a2"
            }
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
    # return True since custom models don't need downloading
    if model == "custom":
        return False

    model_cache_path = Path(".cache/whisper")
    model_path = model_cache_path / f"{model}-ct2"
    if compute_type in ["float16", "int8_float16", "int16", "int8"]:
        compute_type = "float16"
        model_path = model_cache_path / f"{model}-ct2-fp16"

    if compute_type not in MODEL_LINKS[model]:
        if compute_type == "float32":
            compute_type = "float16"
            model_path = model_cache_path / f"{model}-ct2-fp16"
        elif compute_type == "float16":
            compute_type = "float32"
            model_path = model_cache_path / f"{model}-ct2"
        else:
            compute_type = list(MODEL_LINKS[model].keys())[0]
            if compute_type == "float32":
                model_path = model_cache_path / f"{model}-ct2"
            elif compute_type == "float16":
                model_path = model_cache_path / f"{model}-ct2-fp16"

    required_files = ["model.bin", "tokenizer.json"]
    if not model_cache_path.exists() or not model_path.exists():
        return True
    for file_name in required_files:
        if not (model_path / file_name).is_file():
            return True

    expected_hashes = MODEL_LINKS[model][compute_type]["file_checksums"]
    actual_hashes = downloader.load_hashes(model_path)

    if not actual_hashes:
        if downloader.check_file_hashes(model_path, expected_hashes):
            return False
        else:
            return True

    for file_name, expected_hash in expected_hashes.items():
        actual_hash = actual_hashes.get(file_name)
        if actual_hash.lower() != expected_hash.lower():
            if downloader.sha256_checksum(model_path / file_name).lower() == expected_hash.lower():
                actual_hashes[file_name] = expected_hash.lower()
            else:
                return True
    return False


def download_model(model: str, compute_type: str = "float32"):
    model_cache_path = Path(".cache/whisper")
    os.makedirs(model_cache_path, exist_ok=True)
    model_path = model_cache_path / f"{model}-ct2"
    if compute_type in ["float16", "int8_float16", "int16", "int8"]:
        compute_type = "float16"
        model_path = model_cache_path / f"{model}-ct2-fp16"

    if compute_type not in MODEL_LINKS[model]:
        if compute_type == "float32":
            compute_type = "float16"
            model_path = model_cache_path / f"{model}-ct2-fp16"
        elif compute_type == "float16":
            compute_type = "float32"
            model_path = model_cache_path / f"{model}-ct2"
        else:
            compute_type = list(MODEL_LINKS[model].keys())[0]
            if compute_type == "float32":
                model_path = model_cache_path / f"{model}-ct2"
            elif compute_type == "float16":
                model_path = model_cache_path / f"{model}-ct2-fp16"

    pretrained_lang_model_file = model_path / "model.bin"
    file_checksums_check_need_dl = False
    hash_checked_file = model_path / "hash_checked"

    if "file_checksums" in MODEL_LINKS[model][compute_type]:
        if not hash_checked_file.is_file():
            file_checksums_check_need_dl = True

    if not model_path.exists() or not pretrained_lang_model_file.is_file() or file_checksums_check_need_dl:
        print("downloading faster-whisper...")
        if not downloader.download_extract(
                MODEL_LINKS[model][compute_type]["urls"],
                str(model_cache_path.resolve()),
                MODEL_LINKS[model][compute_type]["checksum"],
                title="Speech 2 Text (faster whisper)"
        ):
            print("Model download failed")
        if file_checksums_check_need_dl:
            downloader.save_hashes(model_path, MODEL_LINKS[model][compute_type]["file_checksums"])

    tokenizer_file = model_path / "tokenizer.json"
    if not tokenizer_file.is_file() and model_path.exists():
        tokenizer_type = "normal"
        if ".en" in model:
            tokenizer_type = "en"
        print("downloading tokenizer...")
        if not downloader.download_extract(
                TOKENIZER_LINKS[tokenizer_type]["urls"],
                str(model_path.resolve()),
                TOKENIZER_LINKS[tokenizer_type]["checksum"],
                title="tokenizer"
        ):
            print("Tokenizer download failed")
    elif not model_path.exists():
        print("no model downloaded for tokenizer.")


class FasterWhisper(metaclass=SingletonMeta):
    model = None
    loaded_model_size = ""
    loaded_settings = {}

    sample_rate = 16000

    transcription_count = 0
    reload_after_transcriptions = 0

    def __init__(self, model: str, device: str = "cpu", compute_type: str = "float32", cpu_threads: int = 0,
                 num_workers: int = 1):
        if self.model is None:
            self.load_model(model, device, compute_type, cpu_threads, num_workers)

    def set_reload_after_transcriptions(self, reload_after_transcriptions: int):
        self.reload_after_transcriptions = reload_after_transcriptions

    def release_model(self):
        print("Reloading model...")
        if self.model is not None:
            if hasattr(self.model, 'model'):
                del self.model.model
            if hasattr(self.model, 'feature_extractor'):
                del self.model.feature_extractor
            if hasattr(self.model, 'hf_tokenizer'):
                del self.model.hf_tokenizer
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        self.load_model(
            self.loaded_settings["model"],
            self.loaded_settings["device"],
            self.loaded_settings["compute_type"],
            self.loaded_settings["cpu_threads"],
            self.loaded_settings["num_workers"],
        )

    def load_model(self, model: str, device: str = "cpu", compute_type: str = "float32", cpu_threads: int = 0,
                   num_workers: int = 1):

        self.loaded_settings = {
            "model": model,
            "device": device,
            "compute_type": compute_type,
            "cpu_threads": cpu_threads,
            "num_workers": num_workers
        }

        model_cache_path = Path(".cache/whisper")
        os.makedirs(model_cache_path, exist_ok=True)
        model_folder_name = model + "-ct2"
        if compute_type == "float16" or compute_type == "int8_float16" or compute_type == "int16" or compute_type == "int8":
            model_folder_name = model + "-ct2-fp16"
        # special case for models that are only available in one precision (as float16 vs float32 showed no difference in large-v3 and distilled versions)
        if model != "custom" and compute_type not in MODEL_LINKS[model]:
            if compute_type == "float32":
                model_folder_name = model + "-ct2-fp16"
            elif compute_type == "float16":
                model_folder_name = model + "-ct2"
            else:
                compute_type = list(MODEL_LINKS[model].keys())[0]
                if compute_type == "float32":
                    model_folder_name = model + "-ct2"
                elif compute_type == "float16":
                    model_folder_name = model + "-ct2-fp16"

        # load user custom model
        if model == "custom":
            model_folder_name = "custom-ct2"

        model_path = Path(model_cache_path / model_folder_name)

        self.loaded_model_size = model

        print(f"faster-whisper {model_folder_name} is Loading to {device} using {compute_type} precision...")

        # temporary fix for large-v3 loading (https://github.com/guillaumekln/faster-whisper/issues/547)
        # @TODO: this is a temporary fix for large-v3
        #n_mels = 80
        #use_tf_tokenizer = False
        #if model == "large-v3":
        #    n_mels = 128

        #self.model = WhisperModel(str(Path(model_path).resolve()), device=device, compute_type=compute_type,
        #                          cpu_threads=cpu_threads, num_workers=num_workers, feature_size=n_mels, use_tf_tokenizer=use_tf_tokenizer)
        self.model = WhisperModel(str(Path(model_path).resolve()), device=device, compute_type=compute_type,
                                  cpu_threads=cpu_threads, num_workers=num_workers)

    def transcribe(self, audio_sample, task, language, condition_on_previous_text,
                   initial_prompt, logprob_threshold, no_speech_threshold, temperature, beam_size,
                   word_timestamps, without_timestamps, patience, length_penalty: float = 1,
                   prompt_reset_on_temperature: float = 0.5, repetition_penalty: float = 1,
                   no_repeat_ngram_size: int = 0, multilingual: bool = False) -> dict:

        # large-v3 fix see https://github.com/SYSTRAN/faster-whisper/issues/777
        compression_ratio_threshold = 2.4
        if "-v3" in self.loaded_model_size:
            compression_ratio_threshold -= .2
            logprob_threshold += .3

        result_segments, audio_info = self.model.transcribe(audio_sample, task=task,
                                                            language=language,
                                                            multilingual=multilingual,
                                                            condition_on_previous_text=condition_on_previous_text,
                                                            prompt_reset_on_temperature=prompt_reset_on_temperature,
                                                            initial_prompt=initial_prompt,
                                                            log_prob_threshold=logprob_threshold,
                                                            no_speech_threshold=no_speech_threshold,
                                                            temperature=temperature,
                                                            beam_size=beam_size,
                                                            word_timestamps=word_timestamps,
                                                            without_timestamps=without_timestamps,
                                                            patience=patience,
                                                            length_penalty=length_penalty,
                                                            repetition_penalty=repetition_penalty,
                                                            no_repeat_ngram_size=no_repeat_ngram_size,
                                                            compression_ratio_threshold=compression_ratio_threshold,
                                                            )

        transcription = ""
        segment_list = []
        for segment in result_segments:
            # large-v3 hallucination improvement by only checking no_speech_threshold
            if settings.GetOption("only_no_speech_threshold_for_segments") and segment.no_speech_prob > no_speech_threshold:
                continue

            #audio_data_numpy_split = audio_sample[int(segment.start * self.sample_rate):
            #                                          int(segment.end * self.sample_rate)]
            segment_list.append({'text': segment.text, 'start': segment.start, 'end': segment.end})
            transcription += segment.text + " "


        transcription = transcription.strip()
        result = {
            'text': transcription,
            'type': task,
            'language': audio_info.language,
            'segments': segment_list,
        }

        #self.transcription_count += 1
        #if self.reload_after_transcriptions > 0 and (self.transcription_count % self.reload_after_transcriptions == 0):
        #    self.release_model()

        return result
