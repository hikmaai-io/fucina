package batch

const adoptedBlockTokens = 256

type ReuseSource string

const (
	ReuseSourceNone          ReuseSource = ""
	ReuseSourceLiveSeq       ReuseSource = "live-seq"
	ReuseSourceGPUPagedBlock ReuseSource = "gpu-paged-block"
	ReuseSourceHostSnapshot  ReuseSource = "host-snapshot"
	ReuseSourceDiskSession   ReuseSource = "disk-session"
)

type ReuseInfo struct {
	ReusedTokens int
	Source       ReuseSource
}

func (r ReuseInfo) Normalized() ReuseInfo {
	if r.ReusedTokens < 0 {
		r.ReusedTokens = 0
	}
	if r.ReusedTokens == 0 {
		r.Source = ReuseSourceNone
	}
	return r
}

type AdmitReuseInfoEngine interface {
	LastAddSeqReuse() ReuseInfo
}

type OpenReuseInfoEngine interface {
	LastOpenSeqReuse() ReuseInfo
}

// AdoptedCachedTokens converts a token-level shared prefix to physically reused
// paged-prefix blocks. Only complete 256-token blocks are skipped.
func AdoptedCachedTokens(sharedPrefixTokens int) int {
	if sharedPrefixTokens <= 0 {
		return 0
	}
	return (sharedPrefixTokens / adoptedBlockTokens) * adoptedBlockTokens
}
