// nng_ffi.rs — safe wrapper around system libnng
use std::ffi::{c_char, c_int, c_void, CStr, CString};

#[repr(transparent)]
pub struct Socket(c_int);

// nng error codes are positive
pub const ETIMEDOUT: i32 = 5;

extern "C" {
    fn nng_pub0_open(s: *mut c_int) -> c_int;
    fn nng_sub0_open(s: *mut c_int) -> c_int;
    fn nng_dial(s: c_int, url: *const c_char, opts: *mut c_void, flags: c_int) -> c_int;
    fn nng_send(s: c_int, data: *const u8, len: usize, flags: c_int) -> c_int;
    fn nng_recvmsg(s: c_int, msg: *mut *mut c_void, flags: c_int) -> c_int;
    fn nng_msg_body(msg: *mut c_void) -> *mut u8;
    fn nng_msg_len(msg: *mut c_void) -> usize;
    fn nng_msg_free(msg: *mut c_void);
    fn nng_close(s: c_int) -> c_int;
    fn nng_strerror(err: c_int) -> *const c_char;
    fn nng_socket_set(s: c_int, opt: *const c_char, val: *const c_void, sz: usize) -> c_int;
    fn nng_socket_set_ms(s: c_int, opt: *const c_char, ms: u64) -> c_int;
}

impl Drop for Socket {
    fn drop(&mut self) { unsafe { nng_close(self.0); } }
}

fn err(rc: c_int) -> String {
    let s = unsafe { CStr::from_ptr(nng_strerror(rc)) };
    s.to_string_lossy().into_owned()
}

pub fn pub_open() -> Result<Socket, String> {
    let mut s: c_int = 0;
    let rc = unsafe { nng_pub0_open(&mut s) };
    if rc == 0 { Ok(Socket(s)) } else { Err(err(rc)) }
}

pub fn sub_open() -> Result<Socket, String> {
    let mut s: c_int = 0;
    let rc = unsafe { nng_sub0_open(&mut s) };
    if rc == 0 { Ok(Socket(s)) } else { Err(err(rc)) }
}

pub fn dial(s: &Socket, url: &str) -> Result<(), String> {
    let c = CString::new(url).unwrap();
    let rc = unsafe { nng_dial(s.0, c.as_ptr(), std::ptr::null_mut(), 0) };
    if rc == 0 { Ok(()) } else { Err(err(rc)) }
}

pub fn subscribe(s: &Socket, topic: &[u8]) -> Result<(), String> {
    let opt = CString::new("sub:subscribe").unwrap();
    let rc = unsafe { nng_socket_set(s.0, opt.as_ptr(), topic.as_ptr() as *const c_void, topic.len()) };
    if rc == 0 { Ok(()) } else { Err(err(rc)) }
}

pub fn set_timeout(s: &Socket, ms: u64) -> Result<(), String> {
    let opt = CString::new("recv-timeout").unwrap();
    let rc = unsafe { nng_socket_set_ms(s.0, opt.as_ptr(), ms) };
    if rc == 0 { Ok(()) } else { Err(err(rc)) }
}

pub fn send(s: &Socket, data: &[u8]) -> Result<(), String> {
    let rc = unsafe { nng_send(s.0, data.as_ptr(), data.len(), 0) };
    if rc == 0 { Ok(()) } else { Err(err(rc)) }
}

/// Returns Ok(data) or Err(error_code) — ETIMEDOUT=5 means timeout
pub fn recv(s: &Socket) -> Result<Vec<u8>, i32> {
    let mut msg: *mut c_void = std::ptr::null_mut();
    let rc = unsafe { nng_recvmsg(s.0, &mut msg, 0) };
    if rc == 0 {
        let ptr = unsafe { nng_msg_body(msg) };
        let len = unsafe { nng_msg_len(msg) };
        let mut data = Vec::with_capacity(len);
        unsafe { data.extend_from_slice(std::slice::from_raw_parts(ptr, len)); }
        unsafe { nng_msg_free(msg); }
        Ok(data)
    } else {
        Err(rc)
    }
}

pub fn err_str(rc: i32) -> String { err(rc) }

pub fn read_str(data: &[u8], offset: usize) -> String {
    let slice = &data[offset..];
    let end = slice.iter().position(|&b| b == 0).unwrap_or(32);
    String::from_utf8_lossy(&slice[..end.min(32)]).trim_end().to_string()
}
