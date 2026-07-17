/* Watermark logic for the .coli_kv resume file — pure, testable (test_kvdisk).
 *
 * Bug (2026-07-17): a turn truncated by NGEN gets appended to .coli_kv like any
 * other; the next session resumes it and the model sees its own dangling half
 * answer. The EOS token is never emitted into hist (spec_decode breaks before
 * emit), so complete turns cannot be recognized by scanning tokens: the engine
 * records completeness at APPEND time instead, as a watermark = the largest
 * record count that ends on a complete turn (header word h[7], file magic
 * COLIKV2). Resume trims to the watermark. */
#ifndef COLIBRI_KVDISK_H
#define COLIBRI_KVDISK_H

/* dopo un append fino a `len` record: turno completo -> il watermark avanza,
 * turno troncato da NGEN -> resta sull'ultimo confine completo.
 * EN: after appending up to `len` records. */
static int kv_wm_append(int wm, int len, int complete){
    if(complete) return len<0?0:len;
    return wm<0?0:wm;
}
/* dopo un truncate a `nrec` record (prefix-mismatch API o reset): il watermark
 * non puo' restare oltre la fine del file. EN: wm may only shrink with the file. */
static int kv_wm_truncate(int wm, int nrec){
    if(nrec<0) nrec=0;
    if(wm<0) wm=0;
    return wm<nrec?wm:nrec;
}
/* record sicuri da riprendere: fino al watermark, mai oltre nrec (header
 * corrotto -> clamp, non crash). EN: records safe to resume. */
static int kv_resume_len(int nrec, int wm){
    if(nrec<0) nrec=0;
    if(wm<0) wm=0;
    return wm<nrec?wm:nrec;
}

#endif
