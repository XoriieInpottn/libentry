#!/usr/bin/env python3

from socketserver import ThreadingMixIn
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

__all__ = [
    'XMLRPCServerMixIn'
]


class ThreadXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


class MultRequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2', '/RPC3')


class XMLRPCServerMixIn:

    def serve_forever(self, addr):
        with ThreadXMLRPCServer(addr, requestHandler=MultRequestHandler, allow_none=True) as server:
            server.register_introspection_functions()
            server.register_multicall_functions()
            server.register_instance(self)
            server.serve_forever()
