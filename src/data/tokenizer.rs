pub fn text_to_indices(input: &str) -> Vec<usize> {
	let mut indices = Vec::new();
	for c in input.chars() {
		indices.push(map(c));
	}
	indices
}

fn map(ch: char) -> usize {
	if ch == ' ' {
		return 26;
	}
	ch.to_ascii_lowercase() as usize - 'a' as usize
}
