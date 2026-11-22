"""Generic host for engine packages: control protocol and the adapter session.

An engine package's adapter is a separate process that Skulk starts, talks to
over a length-prefixed JSON control channel and stops. Core never imports
package code. The protocol module holds the message contract; the adapter
module owns one adapter process and its lifetime.
"""
