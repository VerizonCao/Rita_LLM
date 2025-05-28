import pysbd


class TextChunkSpliter:
    """
    Split a streaming text into symmantically independent chunks.
    """

    def __init__(self, language="en"):
        self.segmenter = pysbd.Segmenter(language=language)
        self.buffer = ""
        # self.break_chars = [
        #     "......",
        #     "...",
        #     ".",
        #     "!",
        #     "?",
        #     "\n",
        # ]  # ',',
        self.unwanted_chars = [
            "*",
            "\n",
            "\r",
            "\t",
            "\\",
        ]

    def _reset_splitter(self):
        self.buffer = ""

    def pre_filter_content(self, content):
        for char in self.unwanted_chars:
            content = content.replace(char, "")
        return content

    def post_filter_segments(self, segments):
        filtered_segments = []
        for segment in segments:
            # Remove leading/trailing whitespace or newlines
            segment = segment.strip()
            # Skip if segment is empty or only contains punctuation/whitespace
            if not (
                segment
                and not all(c in " \n\t.,!?;:'\"}{" for c in segment)
                and len(segment) >= 1
            ):
                continue
            # remove right trailing } and " from json
            if segment.endswith("}"):
                segment = segment[:-1].strip()
                if segment.endswith('"'):
                    segment = segment[:-1].strip()

            filtered_segments.append(segment)
        return filtered_segments

    def process_chunk(self, content):
        content = self.pre_filter_content(content)
        self.buffer += content
        segments = self.segmenter.segment(self.buffer)
        if len(segments) > 1:
            self.buffer = segments[-1]
            return self.post_filter_segments(segments[:-1])
        return []

    def get_remaining_buffer(self):
        # return self.post_filter_segments([self.buffer])
        segments_to_return = self.post_filter_segments(
            self.segmenter.segment(self.buffer)
        )
        self._reset_splitter()
        return segments_to_return
