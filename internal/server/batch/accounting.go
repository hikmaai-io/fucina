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

// ResidentCachedPrefix validates that resident is a strict token prefix of prompt.
// physical is the exact already-committed frontier used to append the suffix;
// reported is conservatively rounded down to complete paged-KV blocks.
func ResidentCachedPrefix(resident, prompt []int32) (physical, reported int) {
	if len(resident) == 0 || len(resident) >= len(prompt) {
		return 0, 0
	}
	for i, token := range resident {
		if prompt[i] != token {
			return 0, 0
		}
	}
	reported = AdoptedCachedTokens(len(resident))
	if reported == 0 {
		return 0, 0
	}
	return len(resident), reported
}
