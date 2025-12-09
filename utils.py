# utils.py
# pip install cryptography pycryptodome
from __future__ import annotations

# utils.py
# 依赖：
#   pip install cryptography pycryptodome

# utils.py
# pip install cryptography pycryptodome

import os
import re
import io
import sys
import json
import time
import uuid
import hmac
import base64
import hashlib
import random
import string
import secrets
import socket
from typing import Optional, Union
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, quote, unquote, urlencode, parse_qs

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5
from Crypto import Random


class Utils:
    """
    通用工具类：包含系统工具、时间、文件、JSON、加解密等功能
    """

    # ================= 基础工具 =================
    @staticmethod
    def now_ts() -> int:
        return int(time.time())

    @staticmethod
    def now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def now_iso(tz_offset_hours: int = 8) -> str:
        tz = timezone(timedelta(hours=tz_offset_hours))
        return datetime.now(tz).isoformat(timespec='seconds')

    @staticmethod
    def ts_to_str(ts: int, fmt: str = "%Y-%m-%d %H:%M:%S", tz_offset_hours: int = 8) -> str:
        tz = timezone(timedelta(hours=tz_offset_hours))
        return datetime.fromtimestamp(ts, tz=tz).strftime(fmt)

    @staticmethod
    def str_to_ts(dt_str: str, fmt: str = "%Y-%m-%d %H:%M:%S", tz_offset_hours: int = 8) -> int:
        tz = timezone(timedelta(hours=tz_offset_hours))
        dt = datetime.strptime(dt_str, fmt).replace(tzinfo=tz)
        return int(dt.timestamp())

    @staticmethod
    def safe_getenv(key: str, default: str = "") -> str:
        val = os.getenv(key, default)
        return val.strip() if isinstance(val, str) else default

    @staticmethod
    def rand_str(n: int = 16) -> str:
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(n))

    @staticmethod
    def sha256_hex(data: str) -> str:
        return hashlib.sha256(data.encode('utf-8')).hexdigest()


    # ================= 计时器 =================
    @staticmethod
    @contextmanager
    def timer(name: str = "block"):
        t0 = time.time()
        try:
            yield
        finally:
            print(f"[{name}] {(time.time() - t0)*1000:.1f} ms")

    FIXED_IV = b"1234567890ABdefg"

    @staticmethod
    def _b64e(b: bytes) -> str:
        return base64.b64encode(b).decode("ascii")

    @staticmethod
    def _b64d(s: str) -> bytes:
        t = s.strip().replace("-", "+").replace("_", "/").replace("\n", "").replace("\r", "").replace(" ", "")
        pad = (-len(t)) % 4
        if pad:
            t += "=" * pad
        return base64.b64decode(t)

    @staticmethod
    def aes_key_from_string(s: str) -> str:
        """
        输入任意字符串 -> 生成 AES-GCM 256-bit 密钥 (Base64)
        """
        # SHA-256 输出固定 32 bytes，正好符合 AES-256 要求
        raw_key = hashlib.sha256(s.encode("utf-8")).digest()
        return base64.b64encode(raw_key).decode("utf-8")
    
    @staticmethod
    def aesgcm_encrypt(plaintext: str, key_b64: str, aad: Optional[bytes] = None) -> str:
        if plaintext is None:
            raise ValueError("plaintext is None")
        key = Utils._b64d(key_b64)
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(Utils.FIXED_IV, plaintext.encode("utf-8"), aad)
        return Utils._b64e(ct)

    @staticmethod
    def aesgcm_decrypt(cipher_b64: str, key_b64: str, aad: Optional[bytes] = None) -> str:
        key = Utils._b64d(key_b64)
        ct = Utils._b64d(cipher_b64)
        aesgcm = AESGCM(key)
        pt = aesgcm.decrypt(Utils.FIXED_IV, ct, aad)
        return pt.decode("utf-8")

    # ================= RSA =================
    KEY_SIZE = 1024

    @staticmethod
    def rsa_gen_keypair(key_size: int = KEY_SIZE):
        rng = Random.new().read
        key = RSA.generate(key_size, randfunc=rng)
        pub_der = key.publickey().export_key(format='DER')
        pri_der = key.export_key(format='DER', pkcs=8)
        return base64.b64encode(pub_der).decode(), base64.b64encode(pri_der).decode()

    @staticmethod
    def _b64_to_pubkey(pub_b64: str) -> RSA.RsaKey:
        der = base64.b64decode(pub_b64)
        return RSA.import_key(der)

    @staticmethod
    def _b64_to_prikey(pri_b64: str) -> RSA.RsaKey:
        der = base64.b64decode(pri_b64.replace(' ', '+'))
        return RSA.import_key(der)

    @staticmethod
    def rsa_encrypt(plain_text: str, public_key_b64: str) -> str:
        pub = Utils._b64_to_pubkey(public_key_b64)
        cipher = PKCS1_v1_5.new(pub)
        ct = cipher.encrypt(plain_text.encode('utf-8'))
        return base64.b64encode(ct).decode()

    @staticmethod
    def rsa_decrypt(cipher_b64: str, private_key_b64: str) -> str:
        pri = Utils._b64_to_prikey(private_key_b64)
        cipher = PKCS1_v1_5.new(pri)
        data = base64.b64decode(cipher_b64.replace(' ', '+'))
        sentinel = b'__DECRYPT_ERROR__'
        pt = cipher.decrypt(data, sentinel)
        if pt == sentinel:
            raise ValueError("解密失败（密钥不匹配或数据损坏）")
        return pt.decode('utf-8')

 # ================= 文件 & JSON =================
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def read_text(path: Union[str, Path], default: str = "") -> str:
        p = Path(path)
        return p.read_text(encoding="utf-8") if p.exists() else default

    @staticmethod
    def write_text(path: Union[str, Path], data: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data, encoding="utf-8")

    @staticmethod
    def json_load(path: Union[str, Path], default=None):
        p = Path(path)
        if not p.exists():
            return default
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def json_dump(path: Union[str, Path], obj, ensure_dir: bool = True):
        p = Path(path)
        if ensure_dir:
            p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    # ================= 字符串 / URL =================
    @staticmethod
    def chunk_text(text: str, size: int = 1200) -> list[str]:
        return [text[i:i + size] for i in range(0, len(text), size)]

    @staticmethod
    def slugify(s: str) -> str:
        s = s.lower()
        s = re.sub(r'[^a-z0-9]+', '-', s)
        s = re.sub(r'-+', '-', s).strip('-')
        return s or "n-a"

    @staticmethod
    def url_encode(s: str) -> str:
        return quote(s, safe='')

    @staticmethod
    def url_decode(s: str) -> str:
        return unquote(s)

    @staticmethod
    def is_valid_url(u: str) -> bool:
        try:
            r = urlparse(u)
            return bool(r.scheme and r.netloc)
        except Exception:
            return False
def safe_getenv(key: str, default: str = "") -> str:
    """安全读取环境变量"""
    val = os.getenv(key, default)
    return val.strip() if isinstance(val, str) else default


def now_ts() -> int:
    """当前Unix时间戳（秒）"""
    return int(time.time())


def rand_str(n: int = 16) -> str:
    """生成随机字符串"""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(n))


def sha256_hex(data: str) -> str:
    """字符串SHA256十六进制"""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def b64_encode(data: bytes) -> str:
    """Base64编码"""
    return base64.b64encode(data).decode('utf-8')


def b64_decode(s: str) -> bytes:
    """Base64解码"""
    return base64.b64decode(s)


def json_pretty(obj) -> str:
    """JSON格式化输出"""
    return json.dumps(obj, ensure_ascii=False, indent=2)


def chunk_text(text: str, size: int = 1200):
    """将长文本分块"""
    return [text[i:i + size] for i in range(0, len(text), size)]


def limit_chars(text: str, max_chars: int = 4000) -> str:
    """限制字符串长度"""
    return text if len(text) <= max_chars else text[:max_chars - 3] + "..."


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_text(path: str | Path, default: str = "") -> str:
    """读取文本文件"""
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else default


def write_text(path: str | Path, data: str):
    """写入文本"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data, encoding="utf-8")


def slugify(s: str) -> str:
    """转为URL安全slug"""
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s or "n-a"


def is_valid_url(u: str) -> bool:
    """判断URL合法性"""
    try:
        r = urlparse(u)
        return bool(r.scheme and r.netloc)
    except Exception:
        return False


@contextmanager
def timer(name: str = "block"):
    """耗时统计"""
    t0 = time.time()
    try:
        yield
    finally:
        print(f"[{name}] {(time.time() - t0)*1000:.1f} ms")


