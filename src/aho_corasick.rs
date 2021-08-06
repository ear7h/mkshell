///! A small Aho-Corasick automaton. I can't use the typical Aho-Corasick
///! crate because it doesn't expose the state publicly.

#[derive(Debug)]
struct Node<K> {
    height: usize,
    links: Vec<(K, usize)>,
}

impl<K> Node<K> {
    fn new(height: usize) -> Self {
        Self {
            height,
            links: Vec::new(),
        }
    }

    fn height(&self) -> usize {
        self.height
    }
}

impl<K> Node<K>
where
    K: Eq,
{
    fn find(&self, k: &K) -> Option<usize> {
        for (k1, next) in self.links.iter() {
            if k == k1 {
                return Some(*next);
            }
        }

        None
    }
}

#[derive(Debug)]
pub struct AhoCorasickBuilder<K, V> {
    values: Vec<(usize, V)>,
    nodes: Vec<Node<K>>,
}

impl<K, V> Default for AhoCorasickBuilder<K, V> {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            nodes: Vec::new(),
        }
    }
}

impl<K, V> AhoCorasickBuilder<K, V>
where
    K: Eq,
{
    fn node_value(&self, idx: usize) -> Option<&V> {
        for (idx1, v) in self.values.iter() {
            if idx == *idx1 {
                return Some(v);
            }
        }

        None
    }

    pub fn insert<I>(&mut self, word: I, val: V)
    where
        I: IntoIterator<Item = K>,
    {
        if self.nodes.is_empty() {
            self.nodes.push(Node::new(0))
        }

        self.insert_rec(1, 0, word.into_iter(), val);
    }

    fn insert_rec<I>(&mut self, height: usize, current: usize, mut word: I, val: V)
    where
        I: Iterator<Item = K>,
    {
        match word.next() {
            None => {
                self.values.push((current, val));
            }
            Some(x) => {
                if let Some(next) = self.nodes[current].find(&x) {
                    self.insert_rec(height + 1, next, word, val);
                } else {
                    let next = self.nodes.len();
                    self.nodes[current].links.push((x, next));
                    self.nodes.push(Node::new(height));
                    self.insert_rec(height + 1, next, word, val);
                }
            }
        }
    }

    fn find(&self, current: usize, word: &[&K]) -> Option<usize> {
        match word {
            [] => None,
            [x] => self.nodes[current].find(x),
            [x, xs @ ..] => self.nodes[current]
                .find(x)
                .and_then(|next| self.find(next, xs)),
        }
    }

    fn dfs<F: FnMut(usize, &[&K])>(&self, mut f: F) {
        let mut word = Vec::new();
        for (k, next) in self.nodes[0].links.iter() {
            word.push(k);
            self.dfs_rec(*next, &mut word, &mut f);
            word.pop();
        }
    }

    fn dfs_rec<'a, F: FnMut(usize, &[&'a K])>(
        &'a self,
        current: usize,
        word: &mut Vec<&'a K>,
        f: &mut F,
    ) {
        f(current, word);
        for (k, next) in self.nodes[current].links.iter() {
            word.push(k);
            self.dfs_rec(*next, word, f);
            word.pop();
        }
    }

    fn is_accept(&self, idx: usize) -> bool {
        for (idx1, _) in self.values.iter() {
            if idx == *idx1 {
                return true;
            }
        }

        false
    }

    pub fn build(self) -> AhoCorasick<K, V>
    where
        K: std::fmt::Debug,
    {
        // failure links
        let mut failures: Vec<usize> = Vec::new();
        failures.resize(self.nodes.len(), 0);

        self.dfs(|current, word| {
            let mut link = 0;

            for n in 1..word.len() {
                match self.find(0, &word[n..]) {
                    Some(n) => {
                        link = n;
                        break;
                    }
                    _ => link = 0,
                }
            }

            failures[current] = link;
        });

        let mut dicts: Vec<Vec<usize>> = Vec::new();
        dicts.resize(self.nodes.len(), Vec::new());

        for (node, _) in self.values.iter() {
            let mut search = *node;

            while search != 0 {
                search = failures[search];

                if self.is_accept(search) {
                    dicts[*node].push(search);
                }

                if !dicts[search].is_empty() {
                    assert!(search != *node);

                    let mut tmp = Vec::new();
                    std::mem::swap(&mut dicts[search], &mut tmp);
                    dicts[*node].extend_from_slice(tmp.as_slice());
                    std::mem::swap(&mut dicts[search], &mut tmp);
                    break;
                }
            }
        }

        let AhoCorasickBuilder { values, nodes } = self;
        AhoCorasick {
            values,
            nodes,
            failures,
            dicts,
            state: 0,
        }
    }
}

// TODO: a scan method or other scan type
// TODO: state as separate type?
// TODO: store values, failures and dictionary links
// in the Node
#[derive(Default, Debug)]
pub struct AhoCorasick<K, V> {
    values: Vec<(usize, V)>,
    nodes: Vec<Node<K>>,
    failures: Vec<usize>,
    dicts: Vec<Vec<usize>>,
    state: usize,
}

impl<K, V> AhoCorasick<K, V>
where
    K: Eq,
{
    fn get_value(&self, vidx: usize) -> Option<&V> {
        for (idx, v) in self.values.iter() {
            if *idx == vidx {
                return Some(v);
            }
        }

        None
    }

    fn is_accept(&self) -> Vec<&V> {
        // let state = self.state;
        if let Some(first) = self.get_value(self.state) {
            let mut buf = vec![first];
            for dict_idx in self.dicts[self.state].iter() {
                buf.push(self.get_value(*dict_idx).unwrap());
            }
            buf
        } else {
            Vec::new()
        }
    }

    pub fn reset(&mut self) {
        self.state = 0;
    }

    /// The size of the biggest possible upcoming match. For example, with
    /// the dictionary `[abc, abcd]` and input `zabcz`, this value would be
    /// `[0, 1, 2, 3, 0]` after each `push` call. Note that there is no
    /// internal buffer that takes up this space, rather this value can
    /// be helpful when using `AhoCorasick` to manipulate a stream.
    pub fn match_buffer_size(&self) -> usize {
        self.nodes[self.state].height
    }

    pub fn matches(&self) -> impl Iterator<Item = &V> {
        self.is_accept().into_iter()
    }

    pub fn push_matches(&mut self, k: &K) -> impl Iterator<Item = &V> {
        self.push(k);
        self.matches()
    }

    pub fn push(&mut self, k: &K) {
        loop {
            if let Some(next) = self.nodes[self.state].find(k) {
                self.state = next;
                break;
            }

            self.state = self.failures[self.state];

            if self.state == 0 {
                if let Some(next) = self.nodes[self.state].find(k) {
                    self.state = next;
                }
                break;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_failure_links() {
        let mut acb = AhoCorasickBuilder::default();

        acb.insert("acc".chars(), 1);
        acb.insert("atc".chars(), 2);
        acb.insert("cat".chars(), 3);
        acb.insert("gcg".chars(), 4);

        // println!("{:?}", acb);

        let ac = acb.build();

        assert_eq!(&ac.failures, &[0, 0, 6, 6, 0, 6, 0, 1, 4, 0, 6, 9]);
    }

    #[test]
    fn test_dict_links() {
        let mut acb = AhoCorasickBuilder::default();

        acb.insert("abc".chars(), 1);
        acb.insert("c".chars(), 2);
        acb.insert("xabc".chars(), 3);

        let mut ac = acb.build();

        assert_eq!(ac.push_matches(&'x').next(), None);
        assert_eq!(ac.match_buffer_size(), 1);

        assert_eq!(ac.push_matches(&'a').next(), None);
        assert_eq!(ac.match_buffer_size(), 2);

        assert_eq!(ac.push_matches(&'b').next(), None);
        assert_eq!(ac.match_buffer_size(), 3);

        assert_eq!(ac.push_matches(&'c').collect(): Vec<_>, vec![&3, &1, &2]);
        assert_eq!(ac.match_buffer_size(), 4);

        assert_eq!(ac.push_matches(&'a').next(), None);
        assert_eq!(ac.match_buffer_size(), 1);

        assert_eq!(ac.push_matches(&'b').next(), None);
        assert_eq!(ac.match_buffer_size(), 2);

        assert_eq!(ac.push_matches(&'c').collect(): Vec<_>, vec![&1, &2]);
        assert_eq!(ac.match_buffer_size(), 3);

        assert_eq!(ac.push_matches(&'z').next(), None);
        assert_eq!(ac.match_buffer_size(), 0);
    }
}
