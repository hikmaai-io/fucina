package batch

const adoptedBlockTokens = 256

// AdoptedCachedTokens converts a token-level shared prefix to physically reused
// paged-prefix blocks. Only complete 256-token blocks are skipped.
func AdoptedCachedTokens(sharedPrefixTokens int) int {
	if sharedPrefixTokens <= 0 {
		return 0
	}
	return (sharedPrefixTokens / adoptedBlockTokens) * adoptedBlockTokens
}
